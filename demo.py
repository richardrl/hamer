# generate data
# python demo.py --out_folder /data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/hamer/demo_out/0812_single_vid_two_people_test --batch_size=6 --meta_folder=/data/pulkitag/models/rli14/data/ego4d_fho/v1/0812_single_vid_two_people_test

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.utils import extract_ego4d_rgb_frame_index
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel
from hamer.utils.render_openpose import render_openpose

import copy
import tqdm
from hamer.datasets.image_dataset import FLIP_KEYPOINT_PERMUTATION
from hamer.datasets.utils import fliplr_keypoints
import time

def main(args):

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))

        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    # this is the PyRenderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png

    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)], key=extract_ego4d_rgb_frame_index)

    # Iterate over all images in folder
    for img_path in tqdm.tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        t0 = time.time()
        det_out = detector(img_cv2)
        print(f"detectron time: {time.time() - t0}")
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue


        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        if args.filter_camera_wearer:
            # TODO: fix this... this is pretty bad
            def get_area_from_bbox(bbox):
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            new_boxes = []
            new_right = []

            # if we have multiple copies of a hand, pick the ones with biggest bounding boxes to associate with the camera wearer
            found_left = np.count_nonzero(right == 0)
            found_right = np.count_nonzero(right == 1)

            if found_left > 0:
                left_boxes = [boxes[_] for _ in np.nonzero(right == 0)[0]]
                left_areas = [get_area_from_bbox(_) for _ in left_boxes]
                left_selected_box = left_boxes[np.argmax(left_areas)]
                new_boxes.append(left_selected_box)
                new_right.append(0)

            if found_right > 0:
                right_boxes = [boxes[_] for _ in np.nonzero(right == 1)[0]]
                right_areas = [get_area_from_bbox(_) for _ in right_boxes]
                right_selected_box = right_boxes[np.argmax(right_areas)]
                new_boxes.append(right_selected_box)
                new_right.append(1)

            boxes = np.stack(new_boxes)
            right = np.stack(new_right)

            assert len(boxes) <= 2
            assert len(right) <= 2

        # Run reconstruction on all detected hands
        # we are creating a dataset for a SINGLE image
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        # dataloader is over all the boxes
        for batch_idx, batch in enumerate(dataloader):
            # we should not detect more than batch size hands
            assert batch_idx < 1, batch_idx
            batch = recursive_to(batch, device)
            with torch.no_grad():
                t0 = time.time()
                out = model(batch)
                tqdm.tqdm.write(f"{time.time() -t0} forward pass time")

            # new code to save hamer results
            out.update({"right": batch['right'],
                        "boxes": batch})

            # save all label data
            torch.save(out, os.path.join(args.out_folder, img_path.name.split(".jpg")[0]+ "_label.torch"))

            multiplier = (2*batch['right']-1)

            # pred_cam contains z, x, y from camera to wrist
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            # TODO: what is this?
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # B, num_keypoints, 2
            pred_keypoints_2d_copy = out['pred_keypoints_2d'].detach().clone()

            # converts keypoints in normalized coordinates to keypoints in pixels
            pred_keypoints_2d_copy = model.cfg.MODEL.IMAGE_SIZE * (pred_keypoints_2d_copy + .5)

            # -> add fake confidence score to the end
            pred_keypoints_2d_copy = torch.cat([pred_keypoints_2d_copy, torch.ones(*pred_keypoints_2d_copy.shape[:2], 1).to(device)], dim=-1)

            if args.render:
                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )
                    # -> H?, W?, 3
                    # pick out all
                    regression_img = render_openpose(regression_img * 255., pred_keypoints_2d_copy[n].data.cpu().numpy()) / 255.
                    # let's place the 2D keypoints ontop of the regression image

                    if args.side_view:
                        side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()

                    # TODO: why do we do this
                    # if it is right hand, identity transform
                    # if it is left hand, multiply b -1. Why?
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                    # Save all meshes to disk
                    if args.save_mesh:
                        camera_translation = cam_t.copy()
                        tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                        tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.render and args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # render the 2d keypoints on top
            for n in range(batch_size):
                pred_kp_np = pred_keypoints_2d_copy[n].data.cpu().numpy().copy()
                new_kp = np.zeros_like(pred_kp_np)
                new_kp[:, 2] = pred_keypoints_2d_copy[n].data.cpu().numpy()[:, 2]

                # undo the affine transform
                aff_t = batch['crop_trans'][n].data.cpu().numpy().copy()

                pred_kp_np[:, :2] = pred_kp_np[:, :2] - aff_t[:, 2]

                pred_kp_np[:, :2] = (np.linalg.inv(aff_t[:, :2]) @ pred_kp_np[:, :2].T).T

                if not batch['right'][n]:
                    pred_kp_np = fliplr_keypoints(pred_kp_np, input_img_overlay.shape[1], FLIP_KEYPOINT_PERMUTATION)
                input_img_overlay = render_openpose(input_img_overlay * 255., pred_kp_np) / 255.

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also. This is a rotated view of the hand with white background.')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument("--meta_folder", type=str, help="Folder containing folders, to be run in sequence")

    parser.add_argument("--filter_camera_wearer", help="Whether or not to take top two hand bounding boxes (ignoring the other two)")
    parser.add_argument("--no_filter_camera_wearer", dest="filter_camera_wearer", action="store_false")
    parser.set_defaults(filter_camera_wearer=False)

    parser.add_argument("--render")
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.set_defaults(render=False)

    args = parser.parse_args()


    if args.meta_folder:
        directories = next(os.walk(args.meta_folder))[1]

        for directory in tqdm.tqdm(directories):
            new_args = copy.deepcopy(args)
            new_args.img_folder = os.path.join(args.meta_folder, directory)
            new_args.out_folder = os.path.join("demo_out", os.path.basename(args.meta_folder.rstrip("/")), directory)
            main(new_args)
    else:
        main(args)

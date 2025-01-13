# generate data
# python demo.py --out_folder /data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/hamer/demo_out/0812_single_vid_two_people_test --batch_size=6 --meta_folder=/data/pulkitag/models/rli14/data/ego4d_fho/v1/0812_single_vid_two_people_test

import sys

# remove the hamer from the sys path so imports work right
# python adds the parent directory as the first syspath
sys.path = sys.path[1:]

# assumes files are labelled by the number
from pathlib import Path
import torch
import argparse
import cv2
import numpy as np

from dataset.ego4d_utils import extract_nobbox_generic
from dataset.hamer_dataset_utils import combine_tensor_dict

import hamer
from hamer.hamer.configs import CACHE_DIR_HAMER
from hamer.hamer.utils.utils_detectron2 import get_area_from_bbox
from hamer.hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.hamer.utils import recursive_to
from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from hamer.hamer.utils.render_openpose import render_openpose


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import copy
import tqdm
from dataset.dataset_utils import FLIP_KEYPOINT_PERMUTATION
from hamer.hamer.datasets.utils import fliplr_keypoints
import time

import os
from tqdm.utils import _unicode, disp_len
import sys
import json
from dataset import ego4d_utils
import glob
import pathlib
import math

_TQDM_STATUS_EVERY_N = 2

if "SLURM_JOB_ID" in os.environ:
    def status_printer(self, file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        self._status_printer_counter = 0
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            self._status_printer_counter += 1
            if self._status_printer_counter % _TQDM_STATUS_EVERY_N == 0:
                len_s = disp_len(s)
                # This is where we've removed the \r for clearer output
                fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n')
                last_len[0] = len_s
        return print_status
    tqdm.tqdm.status_printer = status_printer

def torch_loadable(filepath):
   try:
       torch.load(filepath)
       return True
   except Exception as e:
       return False

def main(args, model, model_cfg, device, detector, cpm):
    # Setup the renderer
    # this is the PyRenderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png

    if args.dataset_type == "dexycb":
        assert args.file_type == ["*.jpg"], args.file_type
    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)], key=lambda path: ego4d_utils.extract_rgb_generic(path, args.dataset_type))

    if args.out_folder:
        # scan the image folder for already generated labels, and skip those imgs apths
        print("Only regenerating clip folders we don't have...")
        label_filepaths = glob.glob(os.path.join(args.out_folder, '*_label.torch'))

        if args.check_label_loadable:
            # for each label filepath, try to load
            # if we cannot load it, then we need to remove it from label filepaths so it will be queued for regeneration
            label_filepaths = [path_str for path_str in label_filepaths if torch_loadable(path_str)]

        label_idxs = [ego4d_utils.extract_label_generic(path_str, args.dataset_type) for path_str in label_filepaths]
        label_set = set(label_idxs)

        # all the images
        img_idxs = [ego4d_utils.extract_rgb_generic(path_str, args.dataset_type) for path_str in img_paths]
        img_set = set(img_idxs)

        assert len(img_set) > 0

        # all the bbox undetected indices
        undetected_filepaths = glob.glob(os.path.join(args.out_folder, '*_nobbox.txt'))
        undetected_set = set([extract_nobbox_generic(path_str, args.dataset_type) for path_str in undetected_filepaths])

        # only run our code on images that don't have a label
        if args.dataset_type == "ego4d":
            img_paths = [pathlib.Path(os.path.join(args.img_folder, os.path.basename(args.img_folder.rstrip("/")) + f"_{img_idx:010d}") + ".jpg") for img_idx in (img_set - label_set - undetected_set)]
        elif args.dataset_type == "egoexo":
            img_paths = [pathlib.Path(os.path.join(args.img_folder, f"{img_idx:06d}") + ".jpg") for img_idx in (img_set - label_set - undetected_set)]
        elif args.dataset_type == "dexycb":
            img_paths = [pathlib.Path(os.path.join(args.img_folder, f"color_{img_idx:06d}.jpg")) for img_idx in (img_set - label_set - undetected_set)]
        else:
            raise NotImplementedError

        print("ln100 img folder")
        print(args.img_folder)
        # assert len(img_paths) > 0

    if args.bbox_json:
        bbox_json_loaded = json.load(open(os.path.join(args.bbox_json, os.path.basename(args.out_folder), "bbox.json")))

    def convert_chilarity_string_to_idx(chilarity_str):
        assert chilarity_str in ["left_hand", "right_hand"]
        return 0 if chilarity_str == "left_hand" else 1

    # Iterate over all images in folder
    for img_path in tqdm.tqdm(img_paths):
        print(f"img_path: {img_path}")
        t0 = time.time()
        img_cv2 = cv2.imread(str(img_path))
        assert not (img_cv2 is None)

        print(f"ln103 img load time {time.time() - t0}")

        # convert bgr to rgb before passing into models
        full_img = img_cv2.copy()[:, :, ::-1]

        if args.bbox_json:
            # TODO: load bboxes here
            bbox_data = bbox_json_loaded[f"{ego4d_utils.extract_rgb_generic(img_path, args.dataset_type):06d}"]
            bboxes = [_[0] for _ in bbox_data]
            is_right = [convert_chilarity_string_to_idx(_[1]) for _ in bbox_data]
        else:
            # Detect humans in image
            t0 = time.time()
            det_out = detector(img_cv2)
            tqdm.tqdm.write(f"detectron time: {time.time() - t0}")

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            t0 = time.time()
            vitposes_out = cpm.predict_pose(
                full_img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )
            print(f"vit pose time: {time.time() - t0}")

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
            print("Found 0 bbox, skipping...")
            # label_save_path = os.path.join(args.out_folder, img_path.name.split(".jpg")[0] + "_nobbox.txt")
            label_save_path = os.path.join(args.out_folder, str(ego4d_utils.extract_rgb_generic(img_path, args.dataset_type)) + "_nobbox.txt")

            with open(label_save_path, 'w') as file:
                time.sleep(.0001)

            time.sleep(.0001)
            assert os.path.exists(label_save_path)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        if args.filter_camera_wearer:
            # TODO: fix this... this is pretty bad
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        # dataloader is over all the boxes

        # we should not have more than one batch, or else the code needs updating...
        # TODO: update the code so we collect all the metadata over the entire dataloader and save label torch one time per img
        # some images have way more than 3 people, so can go over the batch size...

        total_out_dict = dict()

        for batch_idx, batch in enumerate(dataloader):
            # we should not detect more than batch size hands
            # assert batch_idx < 1, batch_idx
            batch = recursive_to(batch, device)
            with torch.no_grad():
                t0 = time.time()
                out = model(batch)
                tqdm.tqdm.write(f"{time.time() -t0} hamer forward pass time")

            # new code to save hamer results
            # out.update({"right": batch['right'],
            #             "boxes": batch['boxes']})
            # get everything except img patch
            # TODO: this only works because the batch contains everything
            out.update({k: v for k, v in batch.items() if k != "img"})

            # each out only has tensors....
            # or dict of tensors
            combine_tensor_dict(out, total_out_dict)

            multiplier = (2*batch['right']-1)

            # pred_cam contains z, x, y from camera to wrist
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            full_img_size = batch["full_img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * full_img_size.max()

            # TODO: what is this?
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, full_img_size, scaled_focal_length).detach().cpu().numpy()

            # B, num_keypoints, 2
            pred_keypoints_2d_copy = out['pred_keypoints_2d'].detach().clone()

            # converts keypoints in normalized coordinates to keypoints in pixels
            pred_keypoints_2d_copy = model.cfg.MODEL.IMAGE_SIZE * (pred_keypoints_2d_copy + .5)

            # -> add fake confidence score to the end
            pred_keypoints_2d_copy = torch.cat([pred_keypoints_2d_copy, torch.ones(*pred_keypoints_2d_copy.shape[:2], 1).to(device)], dim=-1)

            if args.render:
                # Render the result
                batch_size = batch['img_patch'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img_patch'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img_patch'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    # produces rgb image
                    # img is the cropped image
                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img_patch'][n],
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

                    # convert rgb back to bgr
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

        # save all label data
        # label_save_path = os.path.join(args.out_folder, img_path.name.split(".jpg")[0]+ "_label.torch")
        label_save_path = os.path.join(args.out_folder, str(ego4d_utils.extract_rgb_generic(img_path, args.dataset_type)) + "_label.torch")

        torch.save(total_out_dict, label_save_path)

        tqdm.tqdm.write(f"Saving: {label_save_path}")
        time.sleep(.001)

        # Render front view
        if args.render and args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # renders the images
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=full_img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            # full image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # render the 2d keypoints on top
            for n in range(batch_size):
                pred_kp_np = pred_keypoints_2d_copy[n].data.cpu().numpy().copy()

                # undo the affine transform
                aff_t = batch['crop_trans'][n].data.cpu().numpy().copy()

                pred_kp_np[:, :2] = pred_kp_np[:, :2] - aff_t[:, 2]

                pred_kp_np[:, :2] = (np.linalg.inv(aff_t[:, :2]) @ pred_kp_np[:, :2].T).T

                if not batch['right'][n]:
                    pred_kp_np = fliplr_keypoints(pred_kp_np, input_img_overlay.shape[1], FLIP_KEYPOINT_PERMUTATION)
                input_img_overlay = render_openpose(input_img_overlay * 255., pred_kp_np) / 255.

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
    print("ln327: demo complete! Predicted hands for all images in the img paths.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images. Assume format: 000723')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results. When passing in a meta folder, this out folder is associated with a clip folder.')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also. This is a rotated view of the hand with white background.')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox. Note: for regnety, should probably be more than 2.0')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument("--meta_folder", type=str, help="Folder containing clip folders, to be run in sequence")

    parser.add_argument("--filter_camera_wearer", help="Whether or not to take top two hand bounding boxes (ignoring the other two)")
    parser.add_argument("--no_filter_camera_wearer", dest="filter_camera_wearer", action="store_false")
    parser.set_defaults(filter_camera_wearer=False)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.add_argument("--mp_idx", type=int, default=None, help="An index to use for multiprocessing.")
    parser.add_argument("--mp_total", type=int, default=None, help="Total number of jobs to use for multiprocessing")
    parser.add_argument("--allowed_parent_tasks", nargs='+', type=str, help="What parent tasks to allow")
    parser.add_argument("--allowed_take_names", nargs='+', type=str, help="Which takes to filter out of the overall take pool")

    # parser.add_argument("--take_name", type=str, help="The name of the subfolder that contains the images")
    parser.add_argument("--bbox_json", type=str, help="Filepath to a precomputed json of bboxes, skips computation of bboxes.")
    parser.add_argument("dataset_type", help="Tells us dataset type which informs image parsing ")
    parser.add_argument("--check_label_loadable", action="store_true", help="When resuming without doing inference, check that each file can be loaded with torch load")
    parser.set_defaults(render=False)

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running with device {device}")
    model = model.to(device)
    model.eval()

    # Load detector
    if args.bbox_json:
        detector = None
    elif args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        # cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        cfg_path = Path("/data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/hamer/hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py")

        detectron2_cfg = LazyConfig.load(str(cfg_path))

        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        raise NotImplementedError

    # keypoint detector
    cpm = ViTPoseModel(device)

    if args.meta_folder:
        # only get chilidren of the meta folder directory
        directories = sorted(next(os.walk(args.meta_folder))[1])

        if args.allowed_take_names:
            valid_take_names = args.allowed_take_names
            directories = [dir_ for dir_ in directories if dir_ in valid_take_names]
        elif args.allowed_parent_tasks:
            # filter to only keep directories with the parent takes
            takes_json_lst = json.load(open(os.path.join("/data/pulkitag/models/rli14/data/egoexo", "takes.json")))
            valid_take_names = [take['take_name'] for take in takes_json_lst if
                                take['parent_task_name'] in args.allowed_parent_tasks]

            directories = [dir_ for dir_ in directories if dir_ in valid_take_names]

        assert len(directories) > 0, "At least one valid directory needed"
        # print("Line 443 temporarily using manual directories!")
        # directories = ['190ae7b6-b295-4dde-9bef-623def96fbab', '1aaa8608-df89-4acf-890c-7e5b87e94f14', '1b2f1101-b3d9-4be5-bc92-393c983be9b4', '1b7f6104-363b-4f1e-8552-e9119a0ae8aa', '1bd06ac3-7ec8-4c44-ba0e-2f88338a913c', '1cee275e-033a-40c7-9709-35abf4606510', '1d37b42e-2a9e-4ba9-aabc-7daac0c15118', '1d468eb0-ff35-47a4-a690-68b4368568b2', '1d75b1a8-0aa9-4dc7-be29-c9f0a30e4d4b', '1f47d622-8691-4a80-aa4e-da6febffc862', '1fd3780c-59b2-4d03-be7c-c46e152f918d', '20756aa3-490c-475f-984c-f87ed0b71816', '827d99e3-ef7a-4b50-b9db-5d6ed65fb051', '8365e83f-cbf3-4260-a76f-565ed04eddf0', '8449f1cb-fd8a-458c-aeb9-b34e2cd7a11d', '857978a5-cab3-4449-928b-6da304c5f7dc', '85ebdb79-0467-440a-b1f8-e48673f1288b', '86421207-534e-4f30-9bfd-d3496493cdf5', '8693853d-e39d-4af7-a702-11ba9129db50', '8709efb8-d4f2-40c8-b34a-8a6524d5e7f6', '87ec3929-5330-4456-9dbb-f42898bf1c23', '8839f088-ddb7-46c8-a897-ef1c017e1495', '88fc0e19-484d-4eda-911e-277ac419c25d', '8918b804-75aa-4bc2-b35b-c22f7fd4a9cd', '8989a6e3-bb75-420e-ac76-cb444b08bdd1', '8a1e4ca7-acc4-450c-87f1-e58952d8452e', '8b9daab7-7739-4dbb-a419-4d451c52df26', '8bcd34ce-09f1-4845-9c38-c56e0ddaf969', '8c61fe0d-36ea-4ab2-b72b-b0971e296187', '8c9ac568-72b6-4749-bdd0-08e13d345614', '8ccb4f1d-d0ce-48a4-8eca-339084de7366', '8d484057-49e2-491c-ae97-c051fa4d06f7', '8d686451-cac9-4526-a022-b6eaf7d467b4', '8f4185b7-9082-4910-b3f3-420bb3a06fd0', '8f861e60-22b7-4ce9-ba75-e647c205f4b2', '905908d5-5d93-4fd4-963a-9df8b16d6fa3', '90621e59-7800-49ab-aeb8-c7725f87a7d8', '91f3ee19-562e-4d39-87ba-b4cb67c4a5f8', '92299954-c927-4e62-b779-ff5a1520fab9', '9267f964-d5a1-4062-872f-657e2a4cbe81', '935a6367-5ebe-482a-8114-74bd46397a63', '93cee4d6-b801-4d41-8710-ae73cc0d040e', '942846d1-d187-4e2e-9c6c-fee3d3c765c5', '94d1b58c-cca3-4509-a217-f30c481ea63b', '9588cee0-77af-4acd-9c9d-f9b6537c9b51', '967d1ca2-d954-43a4-9e1d-2e955dfcc88f', '9683d857-148e-4086-ab24-e6d98bb37b4d', '971bac7a-1a7d-4b48-9177-d992137a7629', '97b457d9-571f-4ce4-8865-157925b23021', '97f3a2e5-fd6c-4982-b0c4-746c728fff0a', '982458ba-0e25-4fec-9b11-d0b378b6e922', '985b27cf-134b-49c8-af60-04881cade20f', '98813290-96b1-4b41-a2f1-802bdf8d26ea', '990258c9-5898-4b06-a046-82f994b82839', '99cda889-b4ce-4c33-b59e-84cf1a69e960', '9b5a0bae-c09b-4921-8468-c86f8dcd8eb7', '9b705436-d0da-41ea-85cf-35903324ad60', '9c20c227-5f30-45c3-8c7c-4d556c24eaed', '9c692f1a-772b-4a22-8601-274c8a2f60ea', '9d6698e7-74d3-4e46-aed3-d43b5fb76095', '9eafa9d7-3825-41d3-a8c9-c00e97699801', '9f39ac52-5e34-4479-9c6f-50d8c153def9', 'a018519a-7aec-4633-9082-4938d165ee76', 'a8f95c5f-7d7b-419b-af29-60d97d8fe379', 'acbc480c-85db-4261-acc5-d8138b492c57', 'accfda52-37f4-491b-bc0b-f0d394ffd6a8', 'add26d3b-1ec7-4bdf-8058-bbe806e56d48', 'aef3563d-4a20-4283-a97b-e51e4836c11d', 'af3391ea-7881-4cde-859d-a779c2edfc4e', 'af828616-e269-4710-9544-75aa87c3995b', 'b1252b5e-97dd-427a-9136-d3966507f6e5', 'b187c2ee-c1fd-4788-80ac-a6e4605b0c70', 'b2277584-8c25-4d49-b7aa-efc6a2d9e8ca', 'b275f09c-5dd2-4e8c-97de-edc1f0c8222b', 'b30b6ca6-7a72-44c0-a224-6986eb2f3fe8', 'b330a68e-3945-4388-a495-7fb467d43089', 'b35de84e-db8c-4c6d-ab33-ad76fc38e19c', 'b40344f8-71b3-4bed-bf99-56a1a9c70f9c', 'b41e3f81-6457-4cd5-be6a-aa0bfccc07e7', 'b5181c11-7fbd-4225-a9db-9ecb56534006', 'b527da99-9e7d-4109-b9eb-7ce483a620a3', 'b54108d9-7542-4a23-928b-a30d5fcb35b9', 'b5bd7bfa-271f-4237-ac6e-f42348adc2a4', 'b5f23940-654a-4796-b071-06b236647e5d', 'b62fc89b-9bd2-4f79-b0cb-b3895a05f454', 'b6585e55-9990-4059-8f01-7dbae41698cb', 'b7922084-d477-4c0b-bd7d-e3852b2fe078', 'b7d315bf-0974-4919-aae9-20e2d5b6cbf1', 'b8cec8d9-b779-427b-8345-3d3ca73ad78a', 'b9f101f3-b778-4737-80db-73e4bd95590e', 'bbf814d6-1a65-4d45-837c-53cafedecb14', 'bc2f466b-bee8-4eab-939c-d3396675b142', 'bd4fdfb8-c1de-44b7-bfe4-3b058cfdeb40', 'bd66979f-2921-420c-98ed-3512e67f79a4', 'bd719ac1-c348-425a-b724-ab27b2b7d6fa', 'bf59f51d-a8c6-4cae-bf4f-415b53fdbbdd', 'c0b017b3-9425-410b-aa3c-8a68990d11bc', 'c2d2f7d4-912e-42b9-bbf5-619a180fef19', 'c3deacc6-9b6c-4250-bf08-466c8c1eaed3', 'c41634ed-20ee-4c2e-9776-47d4445f0f1e', 'c4c0d3fb-8415-43cd-8a07-8d68c6cfcd15', 'dc083d2a-1a4a-41a6-b9dd-c3ca19801708', 'dc5b3dd7-687c-45ac-9c1d-462d07257e66', 'dce8a3ce-a461-448b-a527-7973b5741785', 'ddd35d9d-ca29-40d0-86dd-c1f433337ad3', 'ded39483-8be8-4f8a-a3de-c86b86fd1e7c', 'df008d15-0b65-4770-906c-d2400bf166b5', 'dfa58f3f-366f-4156-9a88-6a2a15fd9bd2', 'dfaa7536-3453-4eab-8d70-f1624d640060', 'e0ea9aa0-acae-428c-a657-cfac402b7fca', 'e127fc34-0de5-41b0-ab68-7d5574bcf613', 'e1d77d87-a7bc-4cc8-b805-e248d4683f8b', 'e1fa5de2-eb21-41a3-85b3-882a3652df42', 'e1ffcfd9-adbc-4a28-aa58-eb6c8189e0fd']


        if args.mp_idx is not None:
            assert args.mp_total
            # split the img paths to be distributed amongst all the different slurm jobs
            # TODO: check this more carefully
            start_img_path_idx = math.floor((len(directories) / args.mp_total) * args.mp_idx)
            end_img_path_idx = math.floor((len(directories) / args.mp_total) * (args.mp_idx + 1))
            directories = directories[start_img_path_idx:end_img_path_idx]

        for directory in tqdm.tqdm(directories):
            new_args = copy.deepcopy(args)
            new_args.img_folder = os.path.join(args.meta_folder, directory)
            # new_args.out_folder = os.path.join(args.out_folder, os.path.basename(args.meta_folder.rstrip("/")), directory)
            new_args.out_folder = os.path.join(args.out_folder, directory)
            main(new_args, model, model_cfg, device, detector, cpm)
    else:
        main(args, model, model_cfg, device, detector, cpm)

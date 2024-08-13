"""
Render OpenPose keypoints.
Code was ported to Python from the official C++ implementation https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/utilities/keypoint.cpp
"""
import cv2
import math
import numpy as np
from typing import List, Tuple

def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    assert keypoints.shape[-1] == 3
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img: np.array,
                     keypoints: np.array,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int32)
                keypoint2 = keypoints[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    return img

def render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        #thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1./50 * map_fn(right_hand_keypoints[:, -1])
    else:
        #thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1./50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20]
    pairs = np.array(pairs).reshape(-1,2)

    colors = [100.,  100.,  100.,
              100.,    0.,    0.,
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1,3)
    #colors = np.zeros_like(colors)
    poseScales = [1]
    #img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(img, right_hand_keypoints, pairs, colors, thicknessCircleRatioRight, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    #img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img

def render_body_keypoints(img: np.array,
                          body_keypoints: np.array) -> np.array:
    """
    Render OpenPose body keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """

    # .25 to make the circles smaller because images are smaller at 320x240
    thickness_circle_ratio = 1./75. * np.ones(body_keypoints.shape[0]) * .25
    thickness_line_ratio_wrt_circle = 0.75
    pairs = []
    pairs = [1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21,11,22,22,23,11,24]
    pairs = np.array(pairs).reshape(-1,2)
    colors = [255.,     0.,     85.,
              255.,     0.,     0.,
              255.,    85.,     0.,
              255.,   170.,     0.,
              255.,   255.,     0.,
              170.,   255.,     0.,
               85.,   255.,     0.,
                0.,   255.,     0.,
              255.,     0.,     0.,
                0.,   255.,    85.,
                0.,   255.,   170.,
                0.,   255.,   255.,
                0.,   170.,   255.,
                0.,    85.,   255.,
                0.,     0.,   255.,
              255.,     0.,   170.,
              170.,     0.,   255.,
              255.,     0.,   255.,
               85.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.]
    colors = np.array(colors).reshape(-1,3)
    pose_scales = [1]
    return render_keypoints(img, body_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)

def render_openpose(img: np.array,
                    hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    #img = render_body_keypoints(img, body_keypoints)
    img = render_hand_keypoints(img, hand_keypoints)
    return img

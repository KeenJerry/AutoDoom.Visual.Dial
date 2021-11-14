import os

import numpy as np
from cv2 import cv2


def float2int(val):
    return tuple(int(x + 0.5) for x in val)


def debug_vis(img, window_corner, label=None, raw_img=None, plot_line=True):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    flag5 = False
    if len(window_corner) == 4:
        left_top, left_bottom, right_bottom, right_top = window_corner
    elif len(window_corner) == 5:
        left_top, left_bottom, right_bottom, right_top, center = window_corner
        flag5 = True
    else:
        assert 0

    num_windows = len(left_top)
    for idx in range(num_windows):
        cv2.putText(cv_img_patch_show, '1', float2int(left_top[idx]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.putText(cv_img_patch_show, '2', float2int(left_bottom[idx]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(cv_img_patch_show, '3', float2int(right_bottom[idx]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(cv_img_patch_show, '4', float2int(right_top[idx]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        cv2.circle(cv_img_patch_show, float2int(left_top[idx]), 3, (255, 0, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(left_bottom[idx]), 3, (0, 255, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(right_bottom[idx]), 3, (0, 0, 255), -1)
        cv2.circle(cv_img_patch_show, float2int(right_top[idx]), 3, (0, 255, 255), -1)

        if flag5:
            cv2.putText(cv_img_patch_show, '5', float2int(center[idx]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.circle(cv_img_patch_show, float2int(center[idx]), 3, (255, 255, 0), -1)

        if plot_line:
            thickness = 2
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(left_top[idx]), float2int(left_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(left_bottom[idx]), float2int(right_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_bottom[idx]), float2int(right_top[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_top[idx]), float2int(left_top[idx]), color, thickness)

    # ----------- vis label --------------
    if isinstance(label, np.ndarray):
        label_ = label.copy() * 255.0
        empty = np.ones((10, cv_img_patch_show.shape[1], 3), dtype=cv_img_patch_show.dtype) * 255
        label_to_draw = np.hstack((label_[0], label_[1], label_[2], label_[3])).astype(cv_img_patch_show.dtype)
        label_to_draw = cv2.cvtColor(label_to_draw, cv2.COLOR_GRAY2BGR)
        cv_img_patch_show = np.vstack((cv_img_patch_show, empty, label_to_draw))

    cv2.imshow('patch', cv_img_patch_show)
    cv2.waitKey(0)

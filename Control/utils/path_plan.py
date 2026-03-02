# native import
import os
import sys
from time import time

# from docker
import cv2 
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryPlanner:
    def __init__(self):
        # camera parameters
        self.K_rgb = np.array([
                [455.2, 0.0, 308.53],
                [0.0, 459.43, 213.56],
                [0.0, 0.0, 1.0],], dtype=np.float32)
        self.tune_height = True
        self.height_adjust = 0.15
        self.d435_height = 0.21
        # line detection parameters
        self.cut_bottom_ratio = 0.10
        self.cut_right_ratio = 0
        self.keep_ratio = 0.8
        self.curve_degree = 2
        self.vertical_ratio_threshold = 3.0
        # line remapping parameters
        self.road_half_width = 0.5/4

    def line_detect(self, image):
        g_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # based on road and  wall intensity  difference, added threshold to recorgnize road -> right bounudary -> extrat outer line. 
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])  # V < 100
        mask = cv2.inRange(hsv_rgb, lower, upper)
        h, w = mask.shape
        cut_row_start = int((1.0 - self.cut_bottom_ratio) * h) # get rid of the  camera mounut at the bottom of the rgb image. 
        cut_col_start = int((1.0 - self.cut_right_ratio) * w)
        mask[cut_row_start:h, :] = 0
        mask[:, cut_col_start:w] = 0

        boundary_points = []
        for y in range(h):
            cols = np.where(mask[y] > 0)[0]
            if cols.size > 0:
                x = int(cols[-1])
                boundary_points.append((x, y))

        line_detected = []
        if self.tune_height:
            h_d435 = self.height_adjust
        else:
            h_d435 = self.d435_height

        for (x, y) in boundary_points:
            x_c, y_c, z_c = self.p2c(x, y, h_d435)
            line_detected.append((x_c, y_c, z_c))

        return line_detected

    def robust_boundary_adjustment(self, x, z,
                                   keep_ratio=0.8,
                                   curve_degree=2,
                                   vertical_ratio_threshold=3.0):

        x = np.asarray(x)
        z = np.asarray(z)

        if len(x) < 3 or len(z) < 3:
            return "insufficient", x, z

        if keep_ratio is None:
            keep_ratio = self.keep_ratio
        if curve_degree is None:
            curve_degree = self.curve_degree
        if vertical_ratio_threshold is None:
            vertical_ratio_threshold = self.vertical_ratio_threshold

        var_x = np.var(x)
        var_z = np.var(z)
        vertical_dominant = var_z > vertical_ratio_threshold * var_x

        n = len(x)
        n_keep = max(2, int(n * keep_ratio))
        n_keep = min(n_keep, n)

        if vertical_dominant:
            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]

            best_width = np.inf
            best_start = 0

            for i in range(n - n_keep + 1):
                width = x_sorted[i + n_keep - 1] - x_sorted[i]
                if width < best_width:
                    best_width = width
                    best_start = i

            x_min = x_sorted[best_start]
            x_max = x_sorted[best_start + n_keep - 1]
            mask = (x >= x_min) & (x <= x_max)

        else:
            sorted_idx = np.argsort(z)
            z_sorted = z[sorted_idx]

            best_width = np.inf
            best_start = 0

            for i in range(n - n_keep + 1):
                width = z_sorted[i + n_keep - 1] - z_sorted[i]
                if width < best_width:
                    best_width = width
                    best_start = i

            z_min = z_sorted[best_start]
            z_max = z_sorted[best_start + n_keep - 1]
            mask = (z >= z_min) & (z <= z_max)

        x_inliers = x[mask]
        z_inliers = z[mask]

        if len(x_inliers) < 2 or len(z_inliers) < 2:
            return "insufficient", x, z

        if vertical_dominant:
            coeffs = np.polyfit(z_inliers, x_inliers, 1)
            x_fit = np.polyval(coeffs, z)
            return "vertical_line", x_fit, z

        dx = np.diff(x_inliers)
        dz = np.diff(z_inliers)
        theta = np.unwrap(np.arctan2(dz, dx))
        std_theta = np.std(theta) if len(theta) > 0 else 0.0

        if std_theta < np.deg2rad(5):
            coeffs = np.polyfit(x_inliers, z_inliers, 1)
            z_fit = np.polyval(coeffs, x)
            return "line", x, z_fit

        # For curve sections, avoid unstable polynomial fitting.
        # Keep filtered boundary points and shift later in target_path().
        curve_sort_idx = np.argsort(z_inliers)
        return "curve", x_inliers[curve_sort_idx], z_inliers[curve_sort_idx]

    def line_detect_adjusted(self, image):
        line_detected = self.line_detect(image)
        if len(line_detected) < 3:
            return "insufficient", np.array([]), np.array([])

        xz = np.array(line_detected, dtype=np.float32)
        x_cam = xz[:, 0]
        z_cam = xz[:, 2]
        _, x_out_adjusted, z_out_adjusted = self.robust_boundary_adjustment(x_cam, z_cam)
        return x_out_adjusted, z_out_adjusted
    
    def target_path(self, x_out_adjusted, z_out_adjusted):
        x_target = x_out_adjusted - self.road_half_width
        z_target = z_out_adjusted
        return x_target, z_target
    
    def line_remap(self, x, z):
        camera_height = self.height_adjust if self.tune_height else self.d435_height
        u, v = self.c2p_ground(x, z, camera_height)
        return u, v

    def p2c(self, u, v, camera_height):
        fx = self.K_rgb[0, 0]
        fy = self.K_rgb[1, 1]
        cx = self.K_rgb[0, 2]
        cy = self.K_rgb[1, 2]
        z = (camera_height * fy) / (v - cy)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y, z

    def c2p_ground(self, x, z, camera_height):
        fx = self.K_rgb[0, 0]
        fy = self.K_rgb[1, 1]
        cx = self.K_rgb[0, 2]
        cy = self.K_rgb[1, 2]

        x = np.asarray(x)
        z = np.asarray(z)

        z_safe = np.where(np.abs(z) < 1e-6, np.nan, z)
        u = fx * x / z_safe + cx
        v = fy * camera_height / z_safe + cy
        return u, v

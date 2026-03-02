#!/usr/bin/env python3
# node written by Haosong to demonstrate qcar2 control
# version 1.1, added non-blocking input for manual control
# 
# Mar 01.2026: v1.3, added visualization and stanley control based on pure computer vision on line detection. 
# CV code is imported from path_plan.py
# ackowledgement: the visualization part of this code is developed with the help of Copilot, need to further improve. 

# native import
import sys
import select
import os
import time
from datetime import datetime

import cv2
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Qcar2 interface
from qcar2_interfaces.msg import MotorCommands

from utils.path_plan import TrajectoryPlanner


class QCar2ControllerNode(Node):

    def __init__(self):
        super().__init__('qcar2_controller_node')

        self.publisher = self.create_publisher(
            MotorCommands,
            'qcar2_motor_speed_cmd',
            10
        )

        self.mode = "AUTO"
        self.steering = 0.0
        self.throttle = 0.0
        self.auto_steering = 0.0
        self.auto_throttle = 0.2
        self.stanley_k = 0.35
        self.stanley_speed = 0.25
        self.stanley_steering_sign = -1.0
        self.auto_max_steer = 0.45
        self.auto_steer_deadband = 0.03
        self.auto_steer_filter_alpha = 0.7

        self.bridge = CvBridge()
        self.trajectory_planner = TrajectoryPlanner()
        self.rgb_subscriber = self.create_subscription(Image, '/camera/color_image', self.image_callback, 10)

        self.save_dir = os.path.join(os.getcwd(), 'captured_images')
        self.auto_overlay_dir = os.path.join(self.save_dir, 'trajectory planned real time')
        os.makedirs(self.auto_overlay_dir, exist_ok=True)

        self.auto_save_interval_sec = 0.25
        self.last_auto_save_time = 0.0
        self.saved_auto_overlay_count = 0
        self.auto_window_name = "AUTO Trajectory Overlay"

        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info("Started in AUTO mode")
        self.get_logger().info("Type command + Enter")
        self.get_logger().info(f"AUTO overlay save directory: {self.auto_overlay_dir}")

    def _compute_stanley(self, x_target, z_target):

        ''' 
            Stanley controller implementation based on the path from pure CV based target trajectory reconstruction.(optimized error control based on copilot suggestions)
            [To-Do 1: We need to further clarify the spacial scale of the track, and consider the spacial realtion of camera w.r.t the vehicle]
            [To-Do 2: Once localization is properly addressed, use ref error as objective function to tune the gains until get best possible aligned trakectory]
        
        '''
        x_target = np.asarray(x_target, dtype=np.float32)
        z_target = np.asarray(z_target, dtype=np.float32)

        valid = np.isfinite(x_target) & np.isfinite(z_target) & (z_target > 0.1)
        x_target = x_target[valid]
        z_target = z_target[valid]

        if len(x_target) < 2:
            return 0.0

        sort_idx = np.argsort(z_target)
        x_sorted = x_target[sort_idx]
        z_sorted = z_target[sort_idx]

        near_mask = (z_sorted >= 0.3) & (z_sorted <= 2.0)
        x_near = x_sorted[near_mask]
        z_near = z_sorted[near_mask]

        if len(x_near) < 2:
            x_near = x_sorted
            z_near = z_sorted

        coeff = np.polyfit(z_near, x_near, 1)  # x = a*z + b
        a = coeff[0]
        b = coeff[1]

        z_ref = 0.8
        heading_error = np.arctan(a)
        cte = a * z_ref + b
        cte_term = np.arctan2(self.stanley_k * cte, self.stanley_speed + 1e-3)

        steering_cmd = self.stanley_steering_sign * (heading_error + cte_term)
        steering_cmd = float(np.clip(steering_cmd, -self.auto_max_steer, self.auto_max_steer))

        if abs(steering_cmd) < self.auto_steer_deadband:
            steering_cmd = 0.0

        steering_cmd = (
            self.auto_steer_filter_alpha * self.auto_steering
            + (1.0 - self.auto_steer_filter_alpha) * steering_cmd
        )

        return steering_cmd

    def image_callback(self, msg):
        if self.mode != "AUTO":
            try:
                cv2.destroyWindow(self.auto_window_name)
            except cv2.error:
                pass
            return

        if msg.width == 0 or msg.height == 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        if cv_image is None or cv_image.size == 0:
            return

        overlay = cv_image.copy()

        try:
            line_detected = self.trajectory_planner.line_detect(cv_image)

            if len(line_detected) > 2:
                xz = np.array(line_detected, dtype=np.float32)
                x_cam = xz[:, 0]
                z_cam = xz[:, 2]

                camera_height = self.trajectory_planner.height_adjust if self.trajectory_planner.tune_height else self.trajectory_planner.d435_height

                def draw_line_from_xz(x_line, z_line, color):
                    u, v = self.trajectory_planner.c2p_ground(x_line, z_line, camera_height)
                    valid = (
                        np.isfinite(u)
                        & np.isfinite(v)
                        & np.isfinite(z_line)
                        & (u >= 0)
                        & (u < overlay.shape[1])
                        & (v >= 0)
                        & (v < overlay.shape[0])
                    )

                    if not np.any(valid):
                        return

                    u_valid = u[valid]
                    v_valid = v[valid]
                    z_valid = z_line[valid]
                    sort_idx = np.argsort(z_valid)
                    uv_points = np.column_stack((u_valid[sort_idx], v_valid[sort_idx])).astype(np.int32)

                    for (u_i, v_i) in uv_points:
                        cv2.circle(overlay, (u_i, v_i), 2, color, -1)

                    if len(uv_points) > 1:
                        cv2.polylines(overlay, [uv_points], isClosed=False, color=color, thickness=2)

                # Raw detected boundary (red)
                draw_line_from_xz(x_cam, z_cam, (0,0,255))

                # Planned/adjusted target path (blue)
                mode, x_adjusted, z_adjusted = self.trajectory_planner.robust_boundary_adjustment(
                    x_cam,
                    z_cam,
                    keep_ratio=self.trajectory_planner.keep_ratio,
                    curve_degree=self.trajectory_planner.curve_degree,
                    vertical_ratio_threshold=self.trajectory_planner.vertical_ratio_threshold,
                )
                if mode != "insufficient":
                    x_target, z_target = self.trajectory_planner.target_path(x_adjusted, z_adjusted)
                    draw_line_from_xz(x_target, z_target, (0, 255, 0))

                cv2.putText(overlay, "Raw: Red  Planned: Green", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)

                if mode != "insufficient":
                    self.auto_steering = self._compute_stanley(x_target, z_target)
                    cv2.putText(overlay, f"AUTO steer: {self.auto_steering:+.2f}", (12, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    self.auto_steering = 0.0

        except Exception as e:
            self.auto_steering = 0.0
            self.get_logger().debug(f"AUTO line overlay skipped: {e}")

        now = time.time()
        if now - self.last_auto_save_time >= self.auto_save_interval_sec:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            save_path = os.path.join(self.auto_overlay_dir, f"auto_traj_overlay_{timestamp}.png")
            if cv2.imwrite(save_path, overlay):
                self.saved_auto_overlay_count += 1
                self.last_auto_save_time = now
                if self.saved_auto_overlay_count % 20 == 0:
                    self.get_logger().info(
                        f"Saved AUTO trajectory overlays: {self.saved_auto_overlay_count}"
                    )

        cv2.imshow(self.auto_window_name, overlay)
        cv2.waitKey(1)

    def loop(self):
        # ---- Publish ----
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']

        if self.mode == "AUTO":
            msg.values = [self.auto_steering, self.auto_throttle]
        else:
            msg.values = [self.steering, self.throttle]

        self.publisher.publish(msg)

        # ---- Non-blocking input ----
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.readline().strip().lower()

            if key == "c":
                self.mode = "MANUAL" if self.mode == "AUTO" else "AUTO"
                print(f"Switched to {self.mode}")

            elif self.mode == "MANUAL":
                if key == "i":
                    self.throttle += 0.2
                elif key == "k":
                    self.throttle -= 0.2
                elif key == "j":
                    self.steering += 0.2
                elif key == "l":
                    self.steering -= 0.2
                elif key == "stop":
                    self.throttle = 0.0
                    self.steering = 0.0

                self.throttle = max(min(self.throttle, 1.0), -1.0)
                self.steering = max(min(self.steering, 1.0), -1.0)


def main(args=None):
    rclpy.init(args=args)
    node = QCar2ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
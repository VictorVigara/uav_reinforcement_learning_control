"""ROS2 node for running the trained RL policy on a real drone.

Subscribes to mocap pose, attitude, and IMU topics. Estimates linear velocity
from position via filtered differentiation. Runs the PPO policy at a fixed
rate and publishes control commands as a Twist message.

Twist mapping:
    linear.x  = thrust        (normalized [-1, 1])
    angular.x = roll rate     (normalized [-1, 1])
    angular.y = pitch rate    (normalized [-1, 1])
    angular.z = yaw rate      (normalized [-1, 1])
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from sensor_msgs.msg import Imu
from stable_baselines3 import PPO

from rl_drone_control.state_estimator import VelocityEstimator
from rl_drone_control.observation_builder import build_observation


class PolicyNode(Node):
    def __init__(self):
        super().__init__("rl_policy_node")

        # ── Parameters ──
        self.declare_parameter("model_path", "")
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 1.0)
        self.declare_parameter("velocity_alpha", 0.8)
        self.declare_parameter("pose_timeout", 0.1)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            self.get_logger().fatal("model_path parameter is required")
            raise RuntimeError("model_path parameter is required")

        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        alpha = self.get_parameter("velocity_alpha").get_parameter_value().double_value
        self._pose_timeout = self.get_parameter("pose_timeout").get_parameter_value().double_value

        # ── Target position (parameter default, overridden by /target_pose topic) ──
        self._target_pos = np.array([
            self.get_parameter("target_x").get_parameter_value().double_value,
            self.get_parameter("target_y").get_parameter_value().double_value,
            self.get_parameter("target_z").get_parameter_value().double_value,
        ])

        # ── Load trained model ──
        self.get_logger().info(f"Loading model from {model_path}...")
        self._model = PPO.load(model_path, device="cpu")
        self.get_logger().info("Model loaded successfully")

        # ── State estimator ──
        self._vel_estimator = VelocityEstimator(alpha=alpha, max_dt=self._pose_timeout)

        # ── Latest sensor data (written by callbacks, read by timer) ──
        self._drone_pos: np.ndarray | None = None
        self._attitude: np.ndarray | None = None
        self._angular_vel: np.ndarray | None = None
        self._linear_vel = np.zeros(3)
        self._last_pose_time: float | None = None
        self._last_imu_time: float | None = None
        self._last_att_time: float | None = None

        # ── Subscribers ──
        self.create_subscription(PoseStamped, "/mocap/pose", self._pose_cb, 10)
        self.create_subscription(Vector3Stamped, "/attitude", self._attitude_cb, 10)
        self.create_subscription(Imu, "/imu", self._imu_cb, 10)
        self.create_subscription(PoseStamped, "/target_pose", self._target_cb, 10)

        # ── Publisher ──
        self._cmd_pub = self.create_publisher(Twist, "/rl_control/cmd", 10)

        # ── Control timer ──
        timer_period = 1.0 / control_rate
        self._timer = self.create_timer(timer_period, self._control_loop)

        # ── Diagnostics throttle ──
        self._diag_counter = 0
        self._diag_interval = int(control_rate)  # log once per second

        self.get_logger().info(
            f"Policy node ready — rate={control_rate}Hz, "
            f"target=[{self._target_pos[0]:.2f}, {self._target_pos[1]:.2f}, {self._target_pos[2]:.2f}]"
        )

    # ── Callbacks ──

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def _pose_cb(self, msg: PoseStamped):
        pos = msg.pose.position
        self._drone_pos = np.array([pos.x, pos.y, pos.z])
        t = self._stamp_to_sec(msg.header.stamp)
        self._linear_vel = self._vel_estimator.update(self._drone_pos, t)
        self._last_pose_time = self.get_clock().now().nanoseconds * 1e-9

    def _attitude_cb(self, msg: Vector3Stamped):
        v = msg.vector
        self._attitude = np.array([v.x, v.y, v.z])  # roll, pitch, yaw
        self._last_att_time = self.get_clock().now().nanoseconds * 1e-9

    def _imu_cb(self, msg: Imu):
        av = msg.angular_velocity
        self._angular_vel = np.array([av.x, av.y, av.z])
        self._last_imu_time = self.get_clock().now().nanoseconds * 1e-9

    def _target_cb(self, msg: PoseStamped):
        pos = msg.pose.position
        self._target_pos = np.array([pos.x, pos.y, pos.z])
        self.get_logger().info(
            f"Target updated: [{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}]"
        )

    # ── Control loop ──

    def _publish_zero(self):
        """Publish zero-thrust safe command."""
        cmd = Twist()
        cmd.linear.x = -1.0  # minimum thrust
        self._cmd_pub.publish(cmd)

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # Safety: check that all sensor data has been received at least once
        if self._drone_pos is None or self._attitude is None or self._angular_vel is None:
            self._publish_zero()
            return

        # Safety: check data freshness
        pose_age = now - self._last_pose_time if self._last_pose_time else float("inf")
        att_age = now - self._last_att_time if self._last_att_time else float("inf")
        imu_age = now - self._last_imu_time if self._last_imu_time else float("inf")

        if pose_age > self._pose_timeout or att_age > self._pose_timeout or imu_age > self._pose_timeout:
            self.get_logger().warn(
                f"Sensor timeout — pose={pose_age:.3f}s, att={att_age:.3f}s, imu={imu_age:.3f}s",
                throttle_duration_sec=1.0,
            )
            self._publish_zero()
            return

        # Build observation
        obs = build_observation(
            drone_pos=self._drone_pos,
            target_pos=self._target_pos,
            attitude=self._attitude,
            linear_vel=self._linear_vel,
            angular_vel=self._angular_vel,
        )

        # Run policy
        action, _ = self._model.predict(obs, deterministic=True)

        # Publish as Twist
        cmd = Twist()
        cmd.linear.x = float(action[0])     # thrust
        cmd.angular.x = float(action[1])     # roll rate
        cmd.angular.y = float(action[2])     # pitch rate
        cmd.angular.z = float(action[3])     # yaw rate
        self._cmd_pub.publish(cmd)

        # Diagnostics (1 Hz)
        self._diag_counter += 1
        if self._diag_counter >= self._diag_interval:
            self._diag_counter = 0
            pos_err = np.linalg.norm(self._target_pos - self._drone_pos)
            self.get_logger().info(
                f"pos_err={pos_err:.3f}m | "
                f"vel=[{self._linear_vel[0]:.2f},{self._linear_vel[1]:.2f},{self._linear_vel[2]:.2f}] | "
                f"act=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}]"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

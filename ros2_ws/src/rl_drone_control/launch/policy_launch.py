"""Launch file for the RL policy inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_path", description="Path to trained .zip model"),
        DeclareLaunchArgument("control_rate", default_value="100.0"),
        DeclareLaunchArgument("target_x", default_value="0.0"),
        DeclareLaunchArgument("target_y", default_value="0.0"),
        DeclareLaunchArgument("target_z", default_value="1.0"),
        DeclareLaunchArgument("velocity_alpha", default_value="0.8"),
        DeclareLaunchArgument("pose_timeout", default_value="0.1"),

        Node(
            package="rl_drone_control",
            executable="policy_node",
            name="rl_policy_node",
            parameters=[{
                "model_path": LaunchConfiguration("model_path"),
                "control_rate": LaunchConfiguration("control_rate"),
                "target_x": LaunchConfiguration("target_x"),
                "target_y": LaunchConfiguration("target_y"),
                "target_z": LaunchConfiguration("target_z"),
                "velocity_alpha": LaunchConfiguration("velocity_alpha"),
                "pose_timeout": LaunchConfiguration("pose_timeout"),
            }],
            output="screen",
        ),
    ])

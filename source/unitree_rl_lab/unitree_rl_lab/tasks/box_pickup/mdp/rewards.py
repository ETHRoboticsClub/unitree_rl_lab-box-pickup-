from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse, quat_apply, quat_from_euler_xyz
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse, quat_rotate as quat_apply, quat_from_euler_xyz
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str | None = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    if command_name is None:
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        try:
            cmd = env.command_manager.get_command(command_name)
            mask = torch.norm(cmd, dim=1) < 0.1
        except KeyError:
            mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return reward * mask


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str | None = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    reward = torch.sum(is_contact, dim=-1).float()
    if command_name is None:
        return reward
    try:
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    except KeyError:
        return reward
    return reward * (command_norm < 0.1)


"""
Hand trajectory rewards (R_traj from paper).
"""


def hand_position_tracking(
    env: ManagerBasedRLEnv,
    left_hand_body_name: str = "left_wrist_roll_rubber_hand",
    right_hand_body_name: str = "right_wrist_roll_rubber_hand",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking hand trajectory positions.
    
    Computes r_hand_pos = ||p_left_hand,t - p*_hand,t_left|| + ||p_right_hand,t - p*_hand,t_right||
    where p*_hand,t are the target hand positions based on the pickup phase.
    
    Target positions:
    - Goal 1 (t=1s=50 steps): Hands 10cm in front of the box's side faces (using box forward direction with yaw)
    - Goal 2 (t=2s=100 steps=t_contact): Hands on side faces of box
    - Goal 3 (t=3.5s=175 steps=t_lift): Hands on either side of target location, spaced by box width
    """
    robot: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene["box"]
    
    # Find hand bodies using exact names from URDF
    try:
        left_hand_id = robot.find_bodies(left_hand_body_name)[0][0]
        right_hand_id = robot.find_bodies(right_hand_body_name)[0][0]
    except (IndexError, KeyError):
        # If we can't find hands, return zero reward
        return torch.zeros(env.num_envs, device=env.device)
    
    left_hand_pos_w = robot.data.body_pos_w[:, left_hand_id]
    right_hand_pos_w = robot.data.body_pos_w[:, right_hand_id]
    
    # Get box properties from command
    command = env.command_manager.get_command("pickup")
    # Command structure: [0:3] dimensions, [3] mass, [4:7] start_pos, [7] start_yaw, [8:11] target_pos, [11] target_yaw, [12] p_contact, [13] p_lift
    # World frame: X=forward (robot facing), Y=left-right, Z=up
    box_size = command[:, :3]  # (num_envs, 3) - [depth (x, forward), width (y, left-right), height (z, up)]
    box_target_pos_rel = command[:, 8:11]  # target box position (relative to env origin)
    box_target_yaw = command[:, 11]  # target yaw
    p_contact = command[:, 12]  # contact phase indicator [0, 1]
    p_lift = command[:, 13]  # lift phase indicator [0, 1]
    
    # Get current box position in world frame
    env_origins = env.scene.env_origins
    box_pos_w = box.data.root_pos_w
    box_quat_w = box.data.root_quat_w
    
    # Get box orientation axes in world frame
    # Box's local frame (when no yaw, aligned with world): 
    #   x = forward (world X), y = left-right (world Y), z = up (world Z)
    # When box has yaw, we rotate these axes using the box's quaternion
    box_right_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)  # Box local y = left-right
    
    box_right = quat_apply(box_quat_w, box_right_local)  # Box's right direction (local y) in world frame
    
    # Box dimensions: [size_x, size_y, size_z] = [depth (forward, X), width (left-right, Y), height (up, Z)]
    box_width = box_size[:, 1:2]  # width (left-right, Y dimension)
    
    # Goal 1 (t=1s=50 steps): Hands 10cm beside the box (using current box orientation)
    # Position: box center ± (half box width + 10cm) to the side
    # Use current box quaternion to determine unit vectors
    goal1_left = box_pos_w - (0.5 * box_width + 0.1) * box_right  # Left side, 10cm from side face
    goal1_right = box_pos_w + (0.5 * box_width + 0.1) * box_right  # Right side, 10cm from side face
    
    # Goal 2 (t=2s=100 steps=t_contact): Hands on side faces of box (using current box orientation)
    # Position: box center ± half box width in Y direction (left-right)
    # Left hand on left side (negative box_right), right hand on right side (positive box_right)
    goal2_left = box_pos_w - 0.5 * box_width * box_right  # Left side
    goal2_right = box_pos_w + 0.5 * box_width * box_right  # Right side
    
    # Goal 3 (t=3.5s=175 steps=t_lift): Hands on either side of target location, spaced by box width
    # Use target yaw to compute unit vectors at target location
    box_target_pos_w = env_origins + box_target_pos_rel
    # Create quaternion from target yaw (yaw-only rotation around Z)
    zero = torch.zeros_like(box_target_yaw)
    target_yaw_quat = quat_from_euler_xyz(zero, zero, box_target_yaw)
    # Get right direction using target yaw
    box_right_target = quat_apply(target_yaw_quat, box_right_local)  # Right direction at target (using target yaw)
    goal3_left = box_target_pos_w - 0.5 * box_width * box_right_target  # Left side
    goal3_right = box_target_pos_w + 0.5 * box_width * box_right_target  # Right side
    
    # Interpolate between goals based on phase (linear interpolation)
    # Before contact (p_contact < 1): interpolate between goal1 and goal2
    # During contact/lift (p_contact >= 1): interpolate between goal2 and goal3 based on p_lift
    alpha_contact = p_contact.clamp(0, 1)
    alpha_lift = p_lift.clamp(0, 1)
    
    # Interpolate goal1 -> goal2 during contact phase
    target_left_contact = goal1_left + alpha_contact.unsqueeze(1) * (goal2_left - goal1_left)
    target_right_contact = goal1_right + alpha_contact.unsqueeze(1) * (goal2_right - goal1_right)
    
    # Interpolate goal2 -> goal3 during lift phase
    target_left = target_left_contact + alpha_lift.unsqueeze(1) * (goal3_left - target_left_contact)
    target_right = target_right_contact + alpha_lift.unsqueeze(1) * (goal3_right - target_right_contact)
    
    # Compute position errors
    error_left = torch.norm(left_hand_pos_w - target_left, dim=1)
    error_right = torch.norm(right_hand_pos_w - target_right, dim=1)
    
    return error_left + error_right


def hand_roll_penalty(
    env: ManagerBasedRLEnv,
    left_hand_body_name: str = "left_wrist_roll_rubber_hand",
    right_hand_body_name: str = "right_wrist_roll_rubber_hand",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize hand roll angles: r_hand_roll = |Φ_left_hand| + |Φ_right_hand|
    
    Encourages hands to maintain flat orientation (no roll).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Find hand bodies using exact names from URDF
    try:
        left_hand_id = robot.find_bodies(left_hand_body_name)[0][0]
        right_hand_id = robot.find_bodies(right_hand_body_name)[0][0]
    except (IndexError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    
    left_hand_quat_w = robot.data.body_quat_w[:, left_hand_id]
    right_hand_quat_w = robot.data.body_quat_w[:, right_hand_id]
    
    # Extract roll angles from quaternions (simplified: use euler angles)
    # For a quaternion (w, x, y, z), roll can be extracted, but we'll use a simpler approximation
    # Roll is rotation around x-axis in body frame
    # We'll compute the angle between hand's x-axis and world's horizontal plane
    
    # Get hand x-axis in world frame
    left_hand_x = quat_apply(left_hand_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    right_hand_x = quat_apply(right_hand_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    
    # Roll angle is the angle between hand x-axis and its projection on horizontal plane
    # Simplified: use z-component as proxy for roll
    left_roll = torch.abs(left_hand_x[:, 2])
    right_roll = torch.abs(right_hand_x[:, 2])
    
    return left_roll + right_roll


"""
Box interaction rewards (R_box from paper).
"""


def box_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_phase_threshold: float = 0.9,
) -> torch.Tensor:
    """Reward for making contact with box at the correct time (during contact phase).
    
    Checks if hands are in contact with box when p_contact is high.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command("pickup")
    p_contact = command[:, 12]  # contact phase indicator
    
    # Check if hands are in contact with box
    # This requires the contact sensor to be configured to detect box contacts
    # For now, we'll use a simplified check based on hand-box distance
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    has_contact = torch.any(is_contact, dim=1).float()
    
    # Reward contact when in contact phase
    in_contact_phase = (p_contact >= contact_phase_threshold).float()
    reward = has_contact * in_contact_phase
    
    return reward


def box_lift_reward(
    env: ManagerBasedRLEnv,
    lift_height_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward for lifting the box (box height increases during lift phase).
    
    Checks if box z-position increases above initial position during lift phase.
    """
    box: RigidObject = env.scene["box"]
    command = env.command_manager.get_command("pickup")
    p_lift = command[:, 13]  # lift phase indicator
    
    # Get initial and current box positions
    env_origins = env.scene.env_origins
    box_initial_pos_rel = command[:, 4:7]  # from command
    box_current_pos_w = box.data.root_pos_w
    box_initial_pos_w = env_origins + box_initial_pos_rel
    
    # Compute height increase
    height_increase = box_current_pos_w[:, 2] - box_initial_pos_w[:, 2]
    
    # Reward lifting during lift phase
    in_lift_phase = (p_lift > 0.0).float()
    lifted = (height_increase > lift_height_threshold).float()
    
    reward = in_lift_phase * lifted * height_increase.clamp(min=0.0)
    
    return reward


def box_target_position_reward(
    env: ManagerBasedRLEnv,
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Reward for moving box to target position (during lift phase).
    
    Checks if box position is close to target position.
    """
    box: RigidObject = env.scene["box"]
    command = env.command_manager.get_command("pickup")
    p_lift = command[:, 13]  # lift phase indicator
    
    # Get target position from command
    env_origins = env.scene.env_origins
    box_target_pos_rel = command[:, 8:11]  # from command
    box_target_pos_w = env_origins + box_target_pos_rel
    
    # Get current box position
    box_current_pos_w = box.data.root_pos_w
    
    # Compute position error
    pos_error = torch.norm(box_current_pos_w - box_target_pos_w, dim=1)
    
    # Reward being close to target during lift phase
    in_lift_phase = (p_lift > 0.5).float()  # Only reward in later part of lift phase
    close_to_target = torch.exp(-pos_error / position_tolerance)
    
    reward = in_lift_phase * close_to_target
    
    return reward


def box_flat_orientation_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward for keeping box in flat orientation (no pitch/roll, only yaw).
    
    Penalizes deviation from flat orientation (box should remain upright).
    """
    box: RigidObject = env.scene["box"]
    box_quat_w = box.data.root_quat_w
    
    # Extract roll and pitch from quaternion
    # For quaternion (w, x, y, z), we can extract euler angles
    # Simplified: check if box's z-axis (up direction) is aligned with world z-axis
    box_up = quat_apply(box_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    
    # Dot product gives cosine of angle between box up and world up
    # For flat orientation, this should be close to 1.0
    alignment = torch.sum(box_up * world_up, dim=1)
    
    # Reward high alignment (box is flat)
    reward = torch.square(alignment)
    
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

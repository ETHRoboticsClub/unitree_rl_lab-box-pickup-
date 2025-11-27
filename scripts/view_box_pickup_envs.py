#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view 100 box pickup environments with G1 robot and boxes.

This script loads 100 parallel environments, each with:
- G1 robot standing upright at xy=(0,0) for each environment
- One box per environment spawned according to BoxRangeCfg specifications
- No gravity
- Environments evenly spaced
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "source"))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="View 100 box pickup environments.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to create.")
# Note: --headless is added by AppLauncher.add_app_launcher_args()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab (which uses Isaac Sim under the hood)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import isaaclab.utils.math as math_utils

# Import tasks to register environments
import unitree_rl_lab.tasks  # noqa: F401

# Import environment config
from unitree_rl_lab.tasks.box_pickup.robots.g1.dof_23.pickup_env_cfg import RobotEnvCfg


def force_robot_upright(robot) -> None:
    """Zero the robot's root orientation."""
    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] = robot.data.root_pos_w
    upright_quat = torch.zeros_like(root_states[:, 3:7])
    upright_quat[:, 0] = 1.0  # (w, x, y, z)
    root_states[:, 3:7] = upright_quat
    root_states[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_states)


def main():
    """Create and visualize 100 box pickup environments."""
    print("=" * 80)
    print("[INFO] Starting box pickup environment viewer")
    print("=" * 80)
    
    # Create environment configuration
    print("[INFO] Creating environment configuration...")
    env_cfg = RobotEnvCfg()
    
    # Set number of environments
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    
    # Disable terrain difficulty curriculum to avoid terrain-level bookkeeping
    if hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
        print("[INFO] Disabled terrain-level curriculum for visualization.")
    
    # Ensure robot spawns at (0, 0) with upright orientation
    # This is already handled by reset_base event, but let's verify
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    print("[INFO] Robot spawn position: (0, 0, 0) with no yaw")
    
    # Set environment spacing for even distribution (drastically reduced for compact grid)
    env_cfg.scene.env_spacing = 1.5  # 0.5 meters between environments (very compact grid)
    print(f"[INFO] Environment spacing: {env_cfg.scene.env_spacing} meters")
    
    # Create environment
    print(f"[INFO] Creating {args_cli.num_envs} environments (this may take a moment)...")
    # Use headless mode if specified (AppLauncher adds this argument)
    render_mode = None if args_cli.headless else "human"
    print(f"[INFO] Render mode: {render_mode if render_mode else 'headless'}")
    
    start_time = time.time()
    env = gym.make("Unitree-G1-23dof-Box-Pickup", cfg=env_cfg, render_mode=render_mode)
    elapsed = time.time() - start_time
    print(f"[INFO] Environment creation completed in {elapsed:.2f} seconds")
    print("[INFO] ✓ Gravity is disabled for both robot and box (configured in env_cfg)")
    
    print(f"[INFO] Environment created with {env.unwrapped.num_envs} parallel environments.")
    print(f"[INFO] Each environment has:")
    print(f"  - G1 robot at xy=(0,0) in its local frame")
    print(f"  - One box spawned according to BoxRangeCfg")
    print(f"  - No gravity")
    print(f"[INFO] Action space: {env.unwrapped.action_space}")
    print(f"[INFO] Observation space: {env.unwrapped.observation_space}")
    
    # Get the correct action dimension from the action manager
    # The action_space.shape might be different from what the action manager expects
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"[INFO] Action dimension (from action manager): {action_dim}")
    if hasattr(env.unwrapped.action_space, 'shape'):
        print(f"[INFO] Action space shape: {env.unwrapped.action_space.shape}")
    
    # Reset environment to initialize all environments
    print("[INFO] Resetting environments (this may take a moment)...")
    start_time = time.time()
    obs, _ = env.reset()
    elapsed = time.time() - start_time
    print(f"[INFO] ✓ Environments reset completed in {elapsed:.2f} seconds")
    print(f"[INFO] Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Debug: Check robot and box positions to verify grid layout
    try:
        robot = env.unwrapped.scene["robot"]
        box = env.unwrapped.scene["box"]
        print(f"[INFO] ✓ Robot found in scene: {type(robot)}")
        print(f"[INFO] ✓ Box found in scene: {type(box)}")
        dims_attr = getattr(env.unwrapped, "box_dimensions", None)
        if dims_attr is not None:
            dims = dims_attr.to("cpu")
            print("[INFO] Box dimensions (first 10 envs):")
            for idx in range(min(10, env.unwrapped.num_envs)):
                sx, sy, sz = dims[idx].tolist()
                print(f"  Env {idx}: size=({sx:.3f}, {sy:.3f}, {sz:.3f})")
        yaw_attr = getattr(env.unwrapped, "box_initial_yaw", None)
        if yaw_attr is not None:
            yaw_deg = yaw_attr[:3] * 180.0 / torch.pi
            print(f"[INFO] Box yaw sample (deg, first 3 envs): {yaw_deg}")
        
        # Get environment origins to verify grid layout
        env_origins = env.unwrapped.scene.env_origins.to(env.unwrapped.device)
        robot_pos = robot.data.root_pos_w
        box_pos = box.data.root_pos_w
        
        print(f"[INFO] Environment origins and positions (first 10 envs):")
        for i in range(min(10, env.unwrapped.num_envs)):
            print(f"  Env {i}:")
            print(f"    Origin: ({env_origins[i, 0]:.3f}, {env_origins[i, 1]:.3f}, {env_origins[i, 2]:.3f})")
            print(f"    Robot:  ({robot_pos[i, 0]:.3f}, {robot_pos[i, 1]:.3f}, {robot_pos[i, 2]:.3f})")
            print(f"    Box:    ({box_pos[i, 0]:.3f}, {box_pos[i, 1]:.3f}, {box_pos[i, 2]:.3f})")
            # Check if robot is at origin
            robot_offset = robot_pos[i] - env_origins[i]
            print(f"    Robot offset from origin: ({robot_offset[0]:.3f}, {robot_offset[1]:.3f}, {robot_offset[2]:.3f})")
    except KeyError as e:
        print(f"[WARNING] Scene entity not found: {e}")
    
    print("=" * 80)
    print("[INFO] Re-aligning robots and boxes to environment origins...")
    print("=" * 80)
    
    # Zero robot orientation once to mimic locomotion viewer behavior
    force_robot_upright(robot)
    
    # Align robots once to their environment origins at (0, 0, base_height)
    device = env.unwrapped.device
    env_origins = env.unwrapped.scene.env_origins.to(device)
    robot_rel = (robot.data.root_pos_w - env_origins).clone()
    robot_rel[:, 0] = 0.0
    robot_rel[:, 1] = 0.0
    robot_locked_pos_w = env_origins + robot_rel
    robot_root_states = robot.data.default_root_state.clone()
    robot_root_states[:, :3] = robot_locked_pos_w
    upright_quat = torch.zeros((env.unwrapped.num_envs, 4), device=device)
    upright_quat[:, 3] = 1.0
    robot_root_states[:, 3:7] = upright_quat
    robot_root_states[:, 7:] = 0.0
    robot.write_root_state_to_sim(robot_root_states)
    
    # Enforce yaw-only orientation for boxes while keeping sampled yaw
    if hasattr(env, "box_initial_pos_rel"):
        box_rel = env.box_initial_pos_rel.to(device=device)
    else:
        box_rel = (box.data.root_pos_w - env_origins).clone()
    box_locked_pos_w = env_origins + box_rel
    box_root_states = box.data.default_root_state.clone()
    box_root_states[:, :3] = box_locked_pos_w
    if hasattr(env, "box_initial_yaw"):
        yaw_vals = env.box_initial_yaw.to(device=device).clone()
        zeros = torch.zeros_like(yaw_vals)
        yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_vals)
    else:
        yaw_quat = box.data.root_quat_w.clone()
    box_root_states[:, 3:7] = yaw_quat / torch.norm(yaw_quat, dim=1, keepdim=True).clamp_min(1e-9)
    box_root_states[:, 7:] = 0.0
    box.write_root_state_to_sim(box_root_states)
    
    print("[INFO] Robot and box alignment completed. Verifying offsets (first 5 envs):")
    for i in range(min(5, env.unwrapped.num_envs)):
        robot_offset = robot_root_states[i, :3] - env_origins[i]
        box_offset = box_root_states[i, :3] - env_origins[i]
        print(f"  Env {i}: robot offset=({robot_offset[0].item():.3f}, {robot_offset[1].item():.3f}, {robot_offset[2].item():.3f}), "
              f"box offset=({box_offset[0].item():.3f}, {box_offset[1].item():.3f}, {box_offset[2].item():.3f})")
    
    print("=" * 80)
    print("[INFO] Environments ready! Starting visualization (no simulation stepping)...")
    print("[INFO] Environments are frozen in their aligned state. Press Ctrl+C to exit.")
    print("=" * 80)
    
    # Simple visualization loop - render without additional state writes
    try:
        while simulation_app.is_running():
            env.unwrapped.scene.update(dt=0.0)
            env.unwrapped.sim.render()
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("[INFO] Keyboard interrupt received. Shutting down...")
        print("=" * 80)
    
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Exception in simulation loop: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
    
    finally:
        # Close environment
        print("[INFO] Closing environment...")
        env.close()
        print("[INFO] ✓ Environment closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close simulation app
        simulation_app.close()

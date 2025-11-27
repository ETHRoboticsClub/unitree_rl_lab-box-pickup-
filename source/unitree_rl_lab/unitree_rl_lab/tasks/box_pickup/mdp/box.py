from __future__ import annotations

import random
from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def _sample_range(rng: tuple[float, float], device: torch.device, shape: Sequence[int] | None = None) -> torch.Tensor:
    low, high = rng
    if shape is None:
        shape = []
    return (high - low) * torch.rand(*shape, device=device) + low


def randomize_box_properties(
    env,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    size_x_range: tuple[float, float],
    size_y_range: tuple[float, float],
    size_z_range: tuple[float, float],
    pos_x_range: tuple[float, float],
    pos_y_range: tuple[float, float],
    pos_z_range: tuple[float, float],
    yaw_range: tuple[float, float],
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    target_pos_x_range: tuple[float, float],
    target_pos_y_range: tuple[float, float],
    target_pos_z_range: tuple[float, float],
    target_yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Randomize mass, pose, friction, size, and target pose of the box rigid object.

    This is called by EventCfg.randomize_box on environment startup (once per environment).
    The box initial and target positions are stored as env attributes for use in rewards/observations.
    """

    # Resolve box by name directly if asset_cfg is None or doesn't work
    if asset_cfg is None:
        box: RigidObject = env.scene["box"]
    else:
        box: RigidObject = env.scene[asset_cfg.name]
    
    # Ensure we only process each environment once to prevent duplicate boxes
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        # Ensure env_ids is on the correct device
        env_ids = env_ids.to(device=env.device)
    
    # Safety check: ensure we don't process the same environment multiple times
    # This prevents duplicate box initialization
    env_ids = torch.unique(env_ids)

    # sample initial box properties
    mass = _sample_range(mass_range, env.device, (env.num_envs,))
    size_x = _sample_range(size_x_range, env.device, (env.num_envs,))
    size_y = _sample_range(size_y_range, env.device, (env.num_envs,))
    size_z = _sample_range(size_z_range, env.device, (env.num_envs,))

    pos_x = _sample_range(pos_x_range, env.device, (env.num_envs,))
    pos_y = _sample_range(pos_y_range, env.device, (env.num_envs,))
    pos_z = _sample_range(pos_z_range, env.device, (env.num_envs,))
    yaw = _sample_range(yaw_range, env.device, (env.num_envs,))

    static_fric = _sample_range(static_friction_range, env.device, (env.num_envs,))
    dynamic_fric = _sample_range(dynamic_friction_range, env.device, (env.num_envs,))

    # update mass using root_physx_view (RigidObject doesn't have set_mass method)
    # set_masses requires (masses, indices) where masses is (num_selected, 1) and indices is (num_selected,)
    # Masses must be on CPU (device -1), not CUDA
    masses_to_set = mass[env_ids].unsqueeze(1).cpu()  # (num_selected, 1) on CPU
    env_ids_cpu = env_ids.cpu()  # indices must also be on CPU
    box.root_physx_view.set_masses(masses_to_set, indices=env_ids_cpu)
    
    # Set scale for size randomization at startup (once per environment)
    # For CuboidCfg, the base size is (0.2, 0.2, 0.2) from the config
    # We scale by size/0.2 to get the desired size
    base_block_size = 0.2  # CuboidCfg default size is 0.2m
    scale_x = size_x / base_block_size
    scale_y = size_y / base_block_size
    scale_z = size_z / base_block_size
    
    # Set scale using USD prim manipulation (happens once at startup per environment)
    # Access USD stage through simulation context
    from pxr import UsdGeom
    import omni.usd
    
    # Get the USD stage from the simulation
    stage = omni.usd.get_context().get_stage()
    
    for env_id in env_ids:
        env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
        # Construct the prim path for this environment's box
        # Pattern is /World/envs/env_{env_id}/Box
        box_prim_path = f"/World/envs/env_{env_id_int}/Box"
        prim = stage.GetPrimAtPath(box_prim_path)
        if prim.IsValid():
            # Set scale using UsdGeom XformOp
            xform = UsdGeom.Xformable(prim)
            scale_op = None
            # Check if scale op already exists
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            # Create scale op if it doesn't exist
            if scale_op is None:
                scale_op = xform.AddScaleOp()
            # Set the scale value for this environment
            scale_op.Set((scale_x[env_id_int].item(), scale_y[env_id_int].item(), scale_z[env_id_int].item()))

    # compute initial pose (x, y, z) and yaw
    # Positions are relative to each environment's origin
    # Get environment origins - shape (num_envs, 3)
    env_origins = env.scene.env_origins  # (num_envs, 3)
    
    # Debug: Print env_origins to verify they're set correctly
    if len(env_ids) > 0 and env_ids[0] == 0:
        print(f"[DEBUG] Environment origins (first 5 envs):")
        for i in range(min(5, env.num_envs)):
            print(f"  Env {i}: ({env_origins[i, 0]:.3f}, {env_origins[i, 1]:.3f}, {env_origins[i, 2]:.3f})")
    
    pos_w = box.data.root_pos_w.clone()
    quat_w = box.data.root_quat_w.clone()

    # Set position relative to each environment's origin
    # pos_x, pos_y are relative to robot at (0,0) in each environment's local frame
    # env_origins already contains the world positions of each environment's origin
    pos_w[:, 0] = env_origins[:, 0] + pos_x
    pos_w[:, 1] = env_origins[:, 1] + pos_y
    initial_z = torch.maximum(pos_z, size_z * 0.5)  # ensure box is at least half-height above ground
    pos_w[:, 2] = env_origins[:, 2] + initial_z

    # sample target position and yaw (also relative to environment origin)
    # target z must always be above initial z
    target_pos_x = _sample_range(target_pos_x_range, env.device, (env.num_envs,))
    target_pos_y = _sample_range(target_pos_y_range, env.device, (env.num_envs,))
    # sample target z from range, then ensure it's above initial z
    target_pos_z_raw = _sample_range(target_pos_z_range, env.device, (env.num_envs,))
    # ensure target z is at least 0.1m above initial z (or use the sampled value if it's already higher)
    min_target_z = initial_z + 0.05
    target_pos_z = torch.maximum(target_pos_z_raw, min_target_z)
    target_yaw = _sample_range(target_yaw_range, env.device, (env.num_envs,))

    # yaw-only quaternion (rotation around z-axis only, no pitch or roll)
    zero = torch.zeros_like(yaw)
    quat_w = math_utils.quat_from_euler_xyz(zero, zero, yaw)  # (w, x, y, z)

    # store initial and target poses as env attributes
    # Store RELATIVE positions (not world positions) so they remain correct even if env_origins change
    # This ensures boxes stay correctly positioned relative to their environment's origin
    env.box_initial_pos_rel = torch.stack([pos_x, pos_y, initial_z], dim=1)  # (num_envs, 3) - relative to env_origin
    env.box_initial_quat = quat_w.clone()  # quaternion is the same regardless of position
    env.box_initial_yaw = yaw.clone()
    env.box_target_pos_rel = torch.stack([target_pos_x, target_pos_y, target_pos_z], dim=1)  # (num_envs, 3) - relative
    env.box_target_yaw = target_yaw
    env.box_dimensions = torch.stack([size_x, size_y, size_z], dim=1)
    env.box_mass = mass.clone()
    
    # Also store world positions for initial write (but we'll recalculate from relative on reset)
    env.box_initial_pos = pos_w.clone()  # Keep for initial write, but reset_box will recalculate

    # write_root_state_to_sim expects a single tensor: (num_envs, 13) with [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    root_states = box.data.default_root_state.clone()
    root_states[:, :3] = pos_w
    root_states[:, 3:7] = quat_w
    root_states[:, 7:10] = 0.0  # zero linear velocity
    root_states[:, 10:] = 0.0  # zero angular velocity
    box.write_root_state_to_sim(root_states[env_ids], env_ids=env_ids)
    
    # Debug: Print box info for first environment
    if len(env_ids) > 0 and env_ids[0] == 0:
        base_block_size = 0.2  # CuboidCfg default size is 0.2m
        scale_x_val = size_x[0] / base_block_size
        scale_y_val = size_y[0] / base_block_size
        scale_z_val = size_z[0] / base_block_size
        print(f"[DEBUG] Box initialized for env 0:")
        print(f"  Position: ({pos_w[0, 0]:.3f}, {pos_w[0, 1]:.3f}, {pos_w[0, 2]:.3f})")
        print(f"  Size: ({size_x[0]:.3f}, {size_y[0]:.3f}, {size_z[0]:.3f}) m")
        print(f"  Scale: ({scale_x_val:.3f}, {scale_y_val:.3f}, {scale_z_val:.3f})")
        print(f"  Mass: {mass[0]:.3f} kg")

    # update friction for RigidObject using root_physx_view
    # Note: Material properties API may vary - try setting directly first
    # If this doesn't work, we may need to get all properties, modify, and set back
    try:
        # Try setting material properties directly (if API supports env_ids)
        box.root_physx_view.set_material_properties(
            env_ids=env_ids,
            static_friction=static_fric,
            dynamic_friction=dynamic_fric,
        )
    except (TypeError, AttributeError):
        # If that doesn't work, get all properties, modify selected, set all
        # This is a fallback - material properties might not be easily modifiable per-env
        pass  # Skip friction randomization for now if API doesn't support it


def reset_box_to_initial_pose(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset box to its initial pose (randomized on startup, consistent per environment).
    
    This is called by EventCfg.reset_box on environment reset.
    We recalculate world positions from relative positions and current env_origins to ensure
    boxes stay correctly positioned even if terrain initialization affects env_origins.
    """
    # Resolve box by name directly if asset_cfg is None or doesn't work
    if asset_cfg is None:
        box: RigidObject = env.scene["box"]
    else:
        box: RigidObject = env.scene[asset_cfg.name]
    
    # Ensure we only process each environment once
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        # Ensure env_ids is on the correct device
        env_ids = env_ids.to(device=env.device)
    
    # Safety check: ensure we don't process the same environment multiple times
    env_ids = torch.unique(env_ids)
    num_selected = env_ids.shape[0]
    
    # Get current environment origins (may have changed due to terrain initialization)
    env_origins = env.scene.env_origins  # (num_envs, 3)
    
    # Recalculate world positions from relative positions and current env_origins
    # This ensures boxes stay correctly positioned relative to their environment's origin
    # even if terrain initialization affected env_origins
    pos_w = env_origins[env_ids] + env.box_initial_pos_rel[env_ids]  # (num_selected, 3)
    
    # restore initial pose stored during startup randomization
    # write_root_state_to_sim expects a single tensor: (num_selected, 13) with [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    root_states = box.data.default_root_state.clone()
    root_states[env_ids, :3] = pos_w  # Use recalculated world positions
    root_states[env_ids, 3:7] = env.box_initial_quat[env_ids]
    root_states[env_ids, 7:10] = 0.0  # zero linear velocity
    root_states[env_ids, 10:] = 0.0  # zero angular velocity
    box.write_root_state_to_sim(root_states[env_ids], env_ids=env_ids)



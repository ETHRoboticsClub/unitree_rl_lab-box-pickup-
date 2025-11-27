from __future__ import annotations

import math
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class BoxPickupCommand(CommandTerm):
    """Command that provides box properties, targets, and phase indicators for the pickup policy."""

    cfg: BoxPickupCommandCfg  # type: ignore[name-defined]

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        # command layout:
        # [0:3] box dimensions (x, y, z)
        # [3]   box mass
        # [4:7] box initial position (relative to env origin)
        # [7]   box initial yaw
        # [8:11] box target position (relative)
        # [11]  box target yaw
        # [12]  contact phase indicator
        # [13]  lift phase indicator
        self._command = torch.zeros(self.num_envs, 14, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        # No additional metrics for now.
        return

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env = self._env
        device = self.device

        def assign(slice_obj, attr_name, default_value=0.0):
            attr = getattr(env, attr_name, None)
            if attr is None:
                self._command[env_ids, slice_obj] = default_value
            else:
                data = attr.to(device=device)
                self._command[env_ids, slice_obj] = data[env_ids]

        assign(slice(0, 3), "box_dimensions")
        assign(3, "box_mass")
        assign(slice(4, 7), "box_initial_pos_rel")
        assign(7, "box_initial_yaw")
        assign(slice(8, 11), "box_target_pos_rel")
        assign(11, "box_target_yaw")
        # reset phases for resampled envs
        self._command[env_ids, 12:] = 0.0

    def _update_command(self):
        # Update phase indicators based on episode steps.
        steps = self._env.episode_length_buf.to(self.device).float()
        contact_steps = max(1, self.cfg.contact_duration_steps)
        lift_steps = max(contact_steps + 1, self.cfg.lift_duration_steps)

        p_contact = torch.clamp(steps / contact_steps, 0.0, 1.0)
        lift_denom = lift_steps - contact_steps
        p_lift = torch.clamp((steps - contact_steps) / max(1, lift_denom), 0.0, 1.0)

        self._command[:, 12] = p_contact
        self._command[:, 13] = p_lift

    def _set_debug_vis_impl(self, debug_vis: bool):
        # No debug visualization.
        raise NotImplementedError


@configclass
class BoxPickupCommandCfg(CommandTermCfg):
    class_type: type = BoxPickupCommand

    contact_duration_steps: int = 100
    lift_duration_steps: int = 175

    def __post_init__(self):
        self.resampling_time_range = (math.inf, math.inf)


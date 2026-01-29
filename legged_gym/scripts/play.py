# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import genesis as gs

import numpy as np
import torch
import time
import sys


def run_simulation(env, policy, logger, robot_index, joint_index, stop_state_log, stop_rew_log):
    """Run simulation in a separate thread using Genesis's run_in_another_thread."""
    obs = env.get_observations()
    
    # Camera offset relative to robot (translation only, no rotation)
    # Position camera behind and above robot to see full robot
    # Use diagonal view: behind, to the side, and above
    camera_offset = np.array([-2.5, 1.5, 2.0])  # Camera position: 2.5m behind, 1.5m to side, 2.0m above robot
    # Lookat point: robot center at typical standing height (~0.3-0.5m for quadruped)
    camera_lookat_height = 0.4  # Height offset for lookat point (robot center height)
    
    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # Update camera to follow robot (translation only, no rotation)
        if hasattr(env.scene, 'viewer') and env.scene.viewer is not None:
            try:
                # Get robot position (only translation, ignore rotation)
                robot_pos = env.base_pos[robot_index].cpu().numpy()
                
                # Camera position is fixed offset from robot position (no rotation applied)
                camera_pos = robot_pos + camera_offset
                
                # Lookat point: robot center at standing height
                camera_lookat = robot_pos + np.array([0.0, 0.0, camera_lookat_height])
                
                # Update viewer camera
                if hasattr(env.scene.viewer, 'set_camera_pose'):
                    env.scene.viewer.set_camera_pose(
                        pos=tuple(camera_pos),
                        lookat=tuple(camera_lookat)
                    )
                else:
                    # Fallback: set camera properties directly
                    env.scene.viewer.camera_pos = tuple(camera_pos)
                    env.scene.viewer.camera_lookat = tuple(camera_lookat)
            except Exception as e:
                pass  # Silently ignore camera update errors
        
        # Logging
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos.get("episode"):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.seed = 999
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    
    # Ensure scene is built
    if not env.scene.is_built:
        env.scene.build(n_envs=env.num_envs)
    
    # Check if viewer exists and we're on macOS
    viewer_exists = hasattr(env.scene, 'viewer') and env.scene.viewer is not None
    is_macos = sys.platform == 'darwin'
    
    # On macOS, use Genesis's separate thread architecture
    if is_macos and not args.headless and viewer_exists:
        print("[INFO] Using Genesis's separate thread architecture for macOS viewer...")
        print("[INFO] Viewer should appear in a separate window.")
        print("[INFO] Press Ctrl+C to stop.")
        
        try:
            # Use Genesis's run_in_another_thread to run simulation in separate thread
            gs.tools.run_in_another_thread(
                fn=run_simulation,
                args=(env, policy, logger, robot_index, joint_index, stop_state_log, stop_rew_log)
            )
            
            # Start viewer in main thread (this blocks until viewer is closed)
            env.scene.viewer.start()
            
        except KeyboardInterrupt:
            print("\n[INFO] Stopping visualization...")
            if hasattr(env.scene.viewer, 'stop'):
                env.scene.viewer.stop()
    else:
        # Non-macOS or headless: use standard loop
        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i == stop_state_log:
                logger.plot_states()
            if 0 < i < stop_rew_log:
                if infos.get("episode"):
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

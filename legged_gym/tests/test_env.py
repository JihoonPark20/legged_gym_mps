# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Smoke test for the Genesis + MPS environment: creates a small number of
# environments and steps them with zero actions.
# Usage: python legged_gym/tests/test_env.py --task go2 --headless

import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def test_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(10 * env.max_episode_length)):
        actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, info = env.step(actions)
    print("Done")


if __name__ == '__main__':
    args = get_args()
    test_env(args)

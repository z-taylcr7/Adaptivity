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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import shutil


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.terrain.measure_heights = True
    env_cfg.env.privileged_obs = True
    env_cfg.env.num_observations = (
        env_cfg.env.num_observations + env_cfg.env.privileged_dim
    )
    train_cfg.runner.run_name = "teacher"

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(env_cfg.env.num_observations)
    ppo_runner, train_cfg = task_registry.make_teacher_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    # train_cfg.policy.net_type = "rma"

    if args.logconfig:
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(LEGGED_GYM_ENVS_DIR, "a1", "a1_config.py"),
            os.path.join(train_cfg.log_dir, "a1_config.py"),
        )
        shutil.copyfile(
            os.path.join(LEGGED_GYM_ENVS_DIR, "base", "legged_robot_config.py"),
            os.path.join(train_cfg.log_dir, "legged_robot_config.py"),
        )
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)

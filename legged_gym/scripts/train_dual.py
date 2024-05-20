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
import json


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.run_name = "dual"
    train_cfg.runner.policy_class_name = "DualActorCritic"
    train_cfg.runner_class_name = "DualPolicyRunner"

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_dual_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    # log_root = os.path.join(
    #     LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    # )

    # log_dir = os.path.join(
    #     log_root,
    #     "dual_" + train_cfg.policy.net_type + "_real",
    #     "y="
    #     + str(train_cfg.policy.num_latent)
    #     + "_l="
    #     + str(train_cfg.policy.history_lengths[1])
    #     + "_"
    #     + str(train_cfg.runner.run_name)
    #     + "_"
    #     + str(datetime.now().strftime("%b%d_%H-%M-%S")),
    # )
    # os.makedirs(log_dir, exist_ok=True)
    # dump_dict_to_json(
    #     vars(train_cfg),
    #     os.path.join(
    #         log_dir,
    #         "train_cfg.json",
    #     ),
    # )
    # dump_dict_to_json(
    #     vars(env_cfg),
    #     os.path.join(
    #         log_dir,
    #         "env_cfg.json",
    #     ),
    # )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


def dump_dict_to_json(dict_obj, file_path):
    with open(file_path, "w") as json_file:
        json.dump(dict_obj, json_file, indent=4)


if __name__ == "__main__":
    args = get_args()
    train(args)

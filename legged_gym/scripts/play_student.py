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
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR


import isaacgym
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    task_registry,
    Logger,
    get_load_path,
)

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    envs_per_label = env_cfg.eval.envs_per_scale
    labels = []
    for vel_x in env_cfg.eval.command_scales_vel_x:
        labels.append(f"cmd_{vel_x}")
    for vel_y in env_cfg.eval.command_scales_vel_y:
        labels.append(f"cmd_y_{vel_y}")
    for fri in env_cfg.eval.friction_scales:
        labels.append(f"fri_{fri}")
    for mas in env_cfg.eval.add_mass_scales:
        labels.append(f"mas_{mas}")
    env_cfg.env.num_envs = min(
        env_cfg.env.num_envs, env_cfg.eval.envs_per_scale * len(labels)
    )
    env_cfg.eval.eval_mode = args.eval_mode != "-1"
    env_cfg.terrain.num_rows = 12
    env_cfg.terrain.num_cols = 12
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.terrain.terrain_proportions = [0.2, 0.4, 0.1, 0.1, 0.2]
    # env_cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0] #smooth slope
    # env_cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0] #rough slope
    # env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0]  # stairs up
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0] #stairs down
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0] #discrete

    # env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.heading = [0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    # train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_student_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    load_run = "rma_student/Ma09_03-47-06_student"
    checkpoint = -1
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
    print(f"Loading model from: {path}")
    ppo_runner.load(path=path, load_student_encoder=True)
    policy, student_encoder = ppo_runner.get_inference_policy_and_encoder(
        device=env.device
    )

    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = train_cfg.policy.history_lengths[1]
    trajectory_history = torch.zeros(
        size=(env_cfg.env.num_envs, history_length, 48), device=student_encoder.device
    )
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], obs[:, :48].unsqueeze(1)), dim=1
    )

    for i in range(10 * int(env.max_episode_length)):
        student_z = student_encoder(trajectory_history)
        concat_obs = torch.concat((obs[:, :48], student_z), dim=-1)
        actions = policy(concat_obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs[:, :48].unsqueeze(1)), dim=1
        )

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            lootat = env.root_states[9, :3]
            # if(int(i / 1000) % 2 == 0):
            #     camara_position = lootat.detach().cpu().numpy() + [0,2,1]
            # else:
            #     camara_position = lootat.detach().cpu().numpy() + [2, 0, 1]
            camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
            env.set_camera(camara_position, lootat)
            # print(env.root_states[0,2])
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        # elif i==stop_state_log:
        #     logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        tracking_errors = env.tracking_errors / (10 * (int(env.max_episode_length)))

    # print(tracking_errors)
    return tracking_errors, labels, envs_per_label


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.eval_mode = "1"
    tracking_errors, labels, envs_per_label = play(args)
    print(labels)
    all_round_errors = {}
    avr_tracking_errors = torch.zeros(len(labels))

    for i in range(len(labels)):
        avr_tracking_errors[i] = (
            torch.sum(tracking_errors[envs_per_label * i : envs_per_label * (i + 1)])
            / envs_per_label
        )
        print(f"test {labels[i]} tracking error: {avr_tracking_errors[i]}")
        all_round_errors[labels[i]] = avr_tracking_errors[i]
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "eval", args.load_run)
    os.makedirs(log_dir, exist_ok=True)
    print("saving to ", log_dir)
    torch.save(avr_tracking_errors, os.path.join(log_dir, "test_tracking_errors.pt"))

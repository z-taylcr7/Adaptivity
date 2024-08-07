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

import argparse
import os
import time

import gym
import numpy as np


import isaacgym
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import (Logger, export_policy_as_jit, get_args,
                              get_load_path, task_registry)

tracking_errors = {}

# envs_per_label = 4
# load_run = "dual_transformer/y=12_l=66_v=3.0_b=2_direct"
# load_run = "dual_transformer/y=12_l=66_v=3.0_direct"
# load_run = "dual_cnn/y=12_l=66_v=3.0"
# load_run = "dual_lstm/y=12_l=66_v=3.0"
# load_run = "dual_gru/y=12_l=66_v=3.0"


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

    env_cfg.eval.eval_mode = args.eval_mode != "0"
    env_cfg.env.num_envs = (
        min(env_cfg.env.num_envs, env_cfg.eval.envs_per_scale * len(labels))
        if env_cfg.eval.eval_mode
        else 10
    )

    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    if not env_cfg.eval.eval_mode:
        env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.ranges.heading = [0.0, 0.0]
        env_cfg.terrain.num_rows = 3
        env_cfg.terrain.num_cols = 4
        env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
        env_cfg.domain_rand.friction_range = [0.5, 0.5]
        env_cfg.domain_rand.randomize_dof_bias = False
        env_cfg.domain_rand.randomize_yaw = False
        env_cfg.domain_rand.randomize_roll = False
        env_cfg.domain_rand.randomize_pitch = False
        env_cfg.domain_rand.randomize_xy = False
        env_cfg.domain_rand.erfi = False
        env_cfg.domain_rand.randomize_obs_latency = False
        env_cfg.domain_rand.randomize_action_latency = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_init_dof = False
        env_cfg.domain_rand.randomize_velo = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_kp_kd = False
        env_cfg.terrain.curriculum = True

    # if terrain_idx == None:
    env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1]
    # env_cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0]
    # else:
    #     env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0]
    #     env_cfg.terrain.terrain_proportions[terrain_idx] = 1
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.policy_class_name = "DualActorCritic"
    train_cfg.runner_class_name = "DualPolicyRunner"
    train_cfg.policy.net_type = train_cfg.runner.load_run.split("/")[0].split("_")[1]
    # train_cfg.policy.net_type = "discrete_transformer"
    if train_cfg.policy.net_type == "teacher":
        env_cfg.terrain.measure_heights = True
        env_cfg.env.privileged_obs = True
        env_cfg.env.num_observations = (
            env_cfg.env.num_observations + env_cfg.env.privileged_dim
        )
    train_cfg.policy.history_lengths = [
        1,
        int(train_cfg.runner.load_run.split("/")[1].split("_")[1].split("=")[1]),
    ]
    # train_cfg.policy.history_lengths = [1,264]
    train_cfg.policy.num_latent = int(
        train_cfg.runner.load_run.split("/")[1].split("_")[0].split("=")[1]
    )
    print("net type:", train_cfg.policy.net_type)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     "./videos",
    #     step_trigger=lambda step: step % 10000
    #     == 0,  # record the videos every 10000 steps
    #     video_length=100,  # for each video record up to 100 steps
    # )
    env.reset()
    obs = env.get_observations()
    # load policy
    # load_run = args.load_run
    if train_cfg.policy.net_type == "teacher":
        ppo_runner, train_cfg = task_registry.make_teacher_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
    else:
        ppo_runner, train_cfg = task_registry.make_dual_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )

    checkpoint = args.checkpoint if args.checkpoint != None else -1
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    path = get_load_path(
        log_root, load_run=train_cfg.runner.load_run, checkpoint=checkpoint
    )
    print(f"Loading model from: {path}")
    ppo_runner.load(path=path)
    policy, encoder = ppo_runner.get_inference_policy_and_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # play

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    eval_laps = 2
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    history_length = train_cfg.policy.history_lengths[1]
    obs_dim = env_cfg.env.num_observations
    trajectory_history = torch.zeros(
        size=(env_cfg.env.num_envs, history_length, obs_dim),
        device=encoder.device,
    )
    print("obs device:", obs.device)
    print("trajectory device:", trajectory_history.device)
    d_obs = obs.detach()
    c_obs = d_obs.to(encoder.device)
    print(c_obs.device)
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], c_obs[:, :obs_dim].unsqueeze(1)),
        dim=1,
    )
    encoder = encoder.to(encoder.device)
    cur_timesteps = torch.zeros(
        (env_cfg.env.num_envs, history_length), dtype=torch.int32, device=encoder.device
    )
    for i in range(eval_laps * int(env.max_episode_length)):
        start = time.time()
        if history_length > 0:
            if train_cfg.policy.net_type == "teacher":
                concat_obs = obs
            else:
                if "transformer" in encoder.net_type:
                    latent = encoder(trajectory_history, cur_timesteps)
                else:
                    latent = encoder(trajectory_history)

                if train_cfg.policy.history_lengths[0] > 0:
                    concat_obs = torch.concat(
                        (
                            trajectory_history[:, -1:, :obs_dim].flatten(1),
                            latent,
                        ),
                        dim=-1,
                    )
                else:
                    concat_obs = latent
        else:
            concat_obs = trajectory_history[:, -1:, :obs_dim].flatten(1)

        concat_obs = concat_obs.to(env.device)
        # print("concat obs device:", concat_obs.device)
        actions = policy(concat_obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        if "transformer" in encoder.net_type:
            cur_timesteps[env_ids] = 0
            cur_timesteps = torch.concat(
                (
                    cur_timesteps[:, 1:],
                    (cur_timesteps[:, -1] + 1).unsqueeze(1),
                ),
                dim=1,
            )
        d_obs = obs.detach()
        c_obs = d_obs.to(encoder.device)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], c_obs[:, :obs_dim].unsqueeze(1)),
            dim=1,
        )
        stop = time.time()
        if i % 1000 == 0:
            print(f"step {i} took {stop - start} seconds")
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
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

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
        elif i == eval_laps * stop_state_log:
            # logger.plot_states()
            pass
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()
        elif i == eval_laps * stop_rew_log:
            break

        # log tracking error

    # print(rews)
    # print(infos)
    # __tracking_errors = np.sqrt(
    #     -0.25
    #     * np.log(np.sum(logger.rew_log["rew_tracking_lin_vel"]) / logger.num_episodes)
    # )
    tracking_errors = env.tracking_errors / (
        eval_laps * (1 + int(env.max_episode_length))
    )

    # print(tracking_errors)
    return tracking_errors, labels, envs_per_label


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    # args.eval_mode = "0"
    EVAL = args.eval_mode != "0"
    print(args.load_run)

    if EVAL:
        tracking_errors, labels, envs_per_label = play(args)
        # print(tracking_errors)
        print(labels)
        all_round_errors = {}
        avr_tracking_errors = torch.zeros(len(labels))

        for i in range(len(labels)):
            avr_tracking_errors[i] = (
                torch.sum(
                    tracking_errors[envs_per_label * i : envs_per_label * (i + 1)]
                )
                / envs_per_label
            )
            print(f"test {labels[i]} tracking error: {avr_tracking_errors[i]}")
            all_round_errors[labels[i]] = avr_tracking_errors[i]
        log_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", "eval_" + args.task, args.load_run
        )
        os.makedirs(log_dir, exist_ok=True)
        print("saving to ", log_dir)
        torch.save(
            avr_tracking_errors, os.path.join(log_dir, "test_tracking_errors.pt")
        )
        # torch.save(
        #     all_round_errors, os.path.join(log_dir, "labelled_tracking_errors.pt")
        # )
    else:
        play(args)
    # test actor MLP variance

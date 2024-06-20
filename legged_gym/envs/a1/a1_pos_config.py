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

from legged_gym.envs.base.legged_robot_config import (LeggedRobotCfg,
                                                      LeggedRobotCfgPPO)


class A1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_observations = 48

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 28.0}  # [N*m/rad]
        damping = {"joint": 0.7}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        pass

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.2
            pass

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.05, 1.75]
        randomize_base_mass = True
        added_mass_range = [-2.0, 5.0]
        randomize_dof_bias = True
        max_dof_bias = 0.08

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0  # not used

        randomize_yaw = True
        init_yaw_range = [-3.14, 3.14]
        randomize_roll = False
        init_roll_range = [-3.14, 3.14]
        randomize_pitch = False
        init_pitch_range = [-3.14, 3.14]
        randomize_xy = True
        init_x_range = [-0.5, 0.5]
        init_y_range = [-0.5, 0.5]
        randomize_velo = True
        init_vlinx_range = [-0.3, 0.3]
        init_vliny_range = [-0.3, 0.3]
        init_vlinz_range = [-0.5, 0.5]
        init_vang_range = [-0.5, 0.5]
        randomize_init_dof = True
        init_dof_factor = [0.5, 1.5]
        stand_bias3 = [0.0, 0.0, 0.0]

        randomize_motor_strength = True  # for a1 flat, enabling this = can not train
        motor_strength_range = [0.9, 1.1]
        randomize_action_latency = True
        randomize_obs_latency = True
        latency_range = [0.005, 0.007]

        erfi = True
        erfi_torq_lim = 7.0 / 9  # per level, curriculum

    class rewards(LeggedRobotCfg.rewards):

        class scales(LeggedRobotCfg.rewards.scales):
            # termination = -2.0
            # torque_limits = -1.0
            dof_vel = -0.00005
            dof_pos_limits = -10.0
            dof_vel_limits = -5.0  # <-0.08
            torques = -0.00005
            fly = -2.0
            # base_height = -0.0
            torque_limits = -0.5
            termination = -10.0
            orientation = -2.0
            collision = -10.0
            feet_collision = -10.0
            # clearance = -0.02
            # slippage = (
            #     -0.002
            # )  # should be less than 0.2. Performance drops when it learns to improve slippage reward

            #     # reach_pos_target_soft = 60.0
            #     # reach_pos_target_tight = 60.0
            #     # reach_heading_target = 30.0
            #     # reach_pos_target_times_heading = 0.0
            #     # velo_dir = 10.0
            #     tracking_lin_vel = 100.0
            #     tracking_ang_vel = 50.0
            #     lin_vel_z = -2.0  #
            #     ang_vel_xy = -0.05

            #     dof_pos_limits = -20.0

            #     lin_vel_z = -2.0
            #     ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            #     action_rate = -0.01
            stand_still = -1.0

        #     # nomove = -20.0

        only_positive_rewards = False

        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25  # 0.25
        foot_height_target = 0.15
        # heading_target_sigma = 1.0
        max_contact_force = 100.0


class A1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_a1"
        max_iterations = 20000
        save_interval = 500

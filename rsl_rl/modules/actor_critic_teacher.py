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

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class TeacherActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        history_lengths,
        net_type,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        latent_dim=12,
        privileged_dim=187 + 16,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(TeacherActorCritic, self).__init__()

        activation = get_activation(activation)

        self.short_history_length = history_lengths[0]
        self.long_history_length = history_lengths[1]

        self.privileged_obs = num_actor_obs
        self.num_actor_obs = num_actor_obs - privileged_dim

        mlp_input_dim_a = (
            self.num_actor_obs
        ) * self.short_history_length + latent_dim  # num_actor_obs=48
        mlp_input_dim_c = (
            self.num_actor_obs
        ) * self.short_history_length + latent_dim  # num_actor_obs=48

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.encoder = Encoder(latent_dim)
        self.privileged_dim = privileged_dim
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        # obs = 48 + 187 + 16
        pr = observations[:, self.num_actor_obs :]
        latent_vec = self.encoder(pr)
        concat_obs = torch.cat(
            (observations[:, : self.num_actor_obs], latent_vec), dim=-1
        )
        self.update_distribution(concat_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent_vector(self, observations):
        latent_vector = self.encoder(
            observations[
                :, self.num_actor_obs : self.num_actor_obs + self.privileged_dim
            ]
        )
        return latent_vector

    def act_inference(self, observations):
        pr = observations[:, self.num_actor_obs :]
        latent_vec = self.encoder(pr)
        concat_obs = torch.cat(
            (observations[:, : self.num_actor_obs], latent_vec), dim=-1
        )
        actions_mean = self.actor(concat_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        pr = critic_observations[:, self.num_actor_obs :]
        latent_vec = self.encoder(pr)
        concat_obs = torch.cat(
            (critic_observations[:, : self.num_actor_obs], latent_vec), dim=-1
        )
        value = self.critic(concat_obs)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim=12,
    ):

        super().__init__()
        self.latent_dim = latent_dim
        self.linear1_1 = nn.Linear(17 * 11, 128)

        self.linear1_2 = nn.Linear(12, 64)
        self.linear1_3 = nn.Linear(4, 64)
        self.relu1 = nn.ELU()

        self.linear2 = nn.Linear(256, 256)
        self.relu2 = nn.ELU()
        self.linear3 = nn.Linear(256, latent_dim)
        self.relu3 = nn.ELU()

    def forward(self, privileged_obs):
        height = privileged_obs[:, : 17 * 11]
        force = privileged_obs[:, 17 * 11 : 17 * 11 + 12]
        paras = privileged_obs[:, 17 * 11 + 12 :]
        x1 = self.linear1_1(height)
        x2 = self.linear1_2(force)
        x3 = self.linear1_3(paras)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x

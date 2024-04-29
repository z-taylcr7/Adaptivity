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
from dual_hist_encoder import DualHistEncoder


class DualActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        history_lengths,
        net_type,
        transformer_direct_act,
        device="cpu",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        num_privileged_obs=187,
        num_latent=64,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(DualActorCritic, self).__init__()
        activation = get_activation(activation)

        self.short_history_length = history_lengths[0]
        self.long_history_length = history_lengths[1]
        self.obs_dim = num_actor_obs
        print(f"Actor obs dim: {self.obs_dim}")
        self.action_dim = num_actions

        self.transformer_direct_act = transformer_direct_act
        self.net_type = net_type

        # self.short_obs_dim = (
        #     num_actor_obs if add_action / 2 else num_actor_obs - num_actions
        # ) - num_privileged_obs
        # self.long_obs_dim = (
        #     num_actor_obs if add_action % 2 else num_actor_obs - num_actions
        # ) - num_privileged_obs

        mlp_input_dim_a = self.obs_dim * self.short_history_length + num_latent * (
            self.long_history_length > 0
        )
        mlp_input_dim_c = mlp_input_dim_a

        self.device = device

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

        print(f"Actor Structure: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # self.encoder = DualHistEncoder(
        #     obs_dim=num_actor_obs,
        #     action_dim=num_actions,
        #     short_history_length=self.short_history_length,
        #     long_history_length=self.long_history_length,
        #     add_action=add_action,
        #     net_type=net_type,
        #     y_dim=num_latent,
        #     device=self.device,
        # ).to(self.device)

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

    def act(self, concat_obs, **kwargs):
        self.update_distribution(concat_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, concat_obs, timestep=-1):
        # latent = None
        # if self.long_history_length > 0:
        #     if self.encoder.net_type == "transformer":
        #         latent = self.encoder(observations[:, :, : self.obs_dim], timestep)
        #     else:
        #         latent = self.encoder(observations[:, :, : self.obs_dim])

        # if self.short_history_length > 0:
        #     short = observations[
        #         :, -self.short_history_length :, : self.obs_dim
        #     ].flatten(1)
        #     if latent is not None:
        #         concat_obs = torch.concat((short, latent), dim=-1)
        #     else:
        #         concat_obs = short
        # else:
        #     concat_obs = latent

        actions_mean = self.actor(concat_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
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

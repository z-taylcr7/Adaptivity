import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal
from transformer_modelling import DecisionTransformer


class DualHistEncoder(nn.Module):
    def __init__(
        self,
        short_history_length=4,
        long_history_length=66,
        obs_dim=48,
        action_dim=12,
        net_type="cnn",
        transformer_direct_act=False,
        device="cpu",
        y_dim=64,
        z_dim=12,
        lr=1e-4,
    ):
        super().__init__()
        self.net_type = net_type
        self.short_history_length = short_history_length
        self.long_history_length = long_history_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.z_dim = z_dim
        self.y_dim = y_dim

        # short_obs_dim = obs_dim if add_action / 2 else obs_dim - action_dim
        # long_obs_dim = obs_dim if add_action % 2 else obs_dim - action_dim

        if self.net_type == "cnn":

            # CNN implementation
            self.conv1 = torch.nn.Conv1d(
                in_channels=obs_dim,
                out_channels=32,
                kernel_size=(8, 1),
                stride=(4, 1),
                padding=0,
                dilation=(1, 1),
                groups=1,
                bias=True,
                padding_mode="zeros",
                device=None,
                dtype=None,
            )
            self.relu1 = nn.ReLU()

            self.conv2 = torch.nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=0,
                dilation=(1, 1),
                groups=1,
                bias=True,
                padding_mode="zeros",
                device=None,
                dtype=None,
            )

            self.relu2 = nn.ReLU()

            self.conv3 = torch.nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=0,
                dilation=(1, 1),
                groups=1,
                bias=True,
                padding_mode="zeros",
                device=None,
                dtype=None,
            )

            self.relu3 = nn.ReLU()

            self.conv4 = torch.nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=0,
                dilation=(1, 1),
                groups=1,
                bias=True,
                padding_mode="zeros",
                device=None,
                dtype=None,
            )

            self.relu4 = nn.ReLU()

            self.linear1 = nn.Linear(96, 32)
            self.relu5 = nn.ReLU()
            self.linear2 = nn.Linear(32, self.y_dim)
            self.relu6 = nn.ReLU()
        elif self.net_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim * self.long_history_length, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, y_dim),
            )
        elif self.net_type == "lstm":
            # LSTM implementation
            self.lstm = nn.LSTM(
                input_size=obs_dim,
                hidden_size=12,
                num_layers=3,
                batch_first=True,
                dropout=0.05,
                # bidirectional=True,
            )
            self.lstm_fc = nn.Linear(self.lstm.hidden_size, self.y_dim)
        elif self.net_type == "rnn":
            self.rnn = nn.RNN(
                input_size=obs_dim,
                hidden_size=12,
                num_layers=3,
                batch_first=True,
                dropout=0.05,
                # bidirectional=True,
            )
            self.rnn_fc = nn.Linear(self.rnn.hidden_size, self.y_dim)
        elif self.net_type == "gru":
            self.gru = nn.GRU(
                input_size=obs_dim,
                hidden_size=12,
                num_layers=3,
                batch_first=True,
                dropout=0.05,
                # bidirectional=True,
            )
            self.gru_fc = nn.Linear(self.gru.hidden_size, self.y_dim)
        elif self.net_type == "transformer":
            self.transformer = DecisionTransformer(
                state_dim=self.obs_dim - self.action_dim,
                act_dim=self.action_dim,
                n_blocks=1,
                h_dim=y_dim,
                context_len=self.long_history_length,
                n_heads=4,
                drop_p=0.05,
                max_timestep=4096,
                # max_timestep=256,
            )
            self.direct_act = transformer_direct_act
            if not self.direct_act:
                self.transformer_fc = nn.Linear(self.action_dim, y_dim)
        elif self.net_type == "discrete_transformer":
            self.discrete_array = [0, 1, 4, 8, 16, 32, 64]
            self.discrete_array = torch.as_tensor(
                self.discrete_array, dtype=torch.long, device=device
            )

            self.transformer = DecisionTransformer(
                state_dim=self.obs_dim - self.action_dim,
                act_dim=self.action_dim,
                n_blocks=1,
                h_dim=y_dim,
                context_len=len(self.discrete_array),
                n_heads=4,
                drop_p=0.05,
                max_timestep=4096,
                # max_timestep=256,
            )
        else:
            raise ValueError("Invalid net_type:", net_type)

        self.optim = Adam(self.parameters(), lr=lr)
        print("DualHistEncoder initialized, obs_dim=", obs_dim)
        print("DualHistEncoder initialized, type=", net_type, "y_dim=", y_dim)
        if self.net_type == "transformer" and self.direct_act:
            print("Direct action if net is transformer!")

    def forward(self, history, cur_timestep=None):
        if self.net_type == "cnn":
            long_history = history
            short_history = history[:, -self.short_history_length :]
            if self.device is not None:
                long_history = torch.as_tensor(
                    long_history,
                    device=self.device,  # type: ignore
                    dtype=torch.float32,
                )
                short_history = torch.as_tensor(
                    short_history,
                    device=self.device,  # type: ignore
                    dtype=torch.float32,
                )

            long_history = long_history.permute(0, 2, 1)
            long_history = long_history.unsqueeze(3)  # for cnn only

            x = self.conv1(long_history)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)

            x = x.flatten(1)
            x = self.linear1(x)
            x = self.relu5(x)
            x = self.linear2(x)
            y = self.relu6(x)
        elif self.net_type == "mlp":
            long_history = torch.as_tensor(
                history.flatten(1),
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            y = self.mlp(long_history.flatten(1))
        elif self.net_type == "lstm":
            long_history = torch.as_tensor(
                history,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            _, (h, c) = self.lstm(long_history)
            y = self.lstm_fc(h[-1])
        elif self.net_type == "rnn":
            long_history = torch.as_tensor(
                history,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            _, h = self.rnn(long_history)
            y = self.rnn_fc(h[-1])
        elif self.net_type == "gru":
            long_history = torch.as_tensor(
                history,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            _, h = self.gru(long_history)
            y = self.gru_fc(h[-1])

        elif self.net_type == "transformer":
            states = history[:, :, : self.obs_dim - self.action_dim]
            actions = history[:, :, self.obs_dim - self.action_dim : self.obs_dim]
            states = torch.as_tensor(
                states,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            actions = torch.as_tensor(
                actions,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )

            timesteps = torch.as_tensor(
                cur_timestep,
                device=self.device,  # type: ignore
                dtype=torch.long,
            )
            y = self.transformer(timesteps, states, actions)
            if not self.direct_act:
                y = self.transformer_fc(y)
        elif self.net_type == "discrete_transformer":
            # discrete_history = torch.zeros(
            #     history.shape[0], len(self.discrete_array), self.obs_dim
            # )
            # discrete_timesteps = torch.zeros(history.shape[0], len(self.discrete_array))
            # for i in range(len(self.discrete_array)):
            #     discrete_history[:, i, :] = history[:, self.discrete_array[i], :]
            #     discrete_timesteps[:, i] = timesteps[:, self.discrete_array[i]]
            timesteps = cur_timestep
            timesteps = timesteps.gather(
                1,
                torch.tensor(self.discrete_array)
                .unsqueeze(0)
                .repeat(timesteps.shape[0], 1),
            )
            discrete_history = history.gather(
                1,
                torch.tensor(self.discrete_array)
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(history.shape[0], 1, history.shape[2]),
            )
            states = discrete_history[:, :, : self.obs_dim - self.action_dim]
            actions = discrete_history[
                :, :, self.obs_dim - self.action_dim : self.obs_dim
            ]

            states = torch.as_tensor(
                states,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            actions = torch.as_tensor(
                actions,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )

            timesteps = torch.as_tensor(
                timesteps,
                device=self.device,  # type: ignore
                dtype=torch.long,
            )
            y = self.transformer(timesteps, states, actions)

        # return torch.randn_like(y) * 0.1 * y + y

        return y

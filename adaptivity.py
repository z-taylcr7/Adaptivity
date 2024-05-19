# load .pt
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import torch
import os

# labels = [
# ver 1.0
#     "cmd_-1.0",  # 0-4
#     "cmd_0.2",  # 5-9 # 0.2,0.2
#     "cmd_0.4",  # 10-14 # 0.4,0.4
#     "cmd_0.8",  # 15-19 # 0.8,0.8
#     "cmd.1.6",  # 20-24
#     "cmd_2.0",  # 25-29
#     "cmd_3.0",  # 30-34
#     "fri_low",  # 35-39  # 0.1
#     "fri_high",  # 40-44  # 2.0
#     "mas",  # 45-49 # 1.0-1.5
# ]
labels = [
    "cmd_-1.5",
    "cmd_-1.0",
    "cmd_-0.5",
    "cmd_0.1",
    "cmd_0.5",
    "cmd_1.0",
    "cmd_1.5",
    "cmd_2.0",
    "cmd_2.5",
    "cmd_3.0",
    "cmd_3.5",
    "cmd_4.0",
    "cmd_y_-1.5",
    "cmd_y_0.0",
    "cmd_y_1.5",
    "fri_0.1",
    "fri_0.5",
    "fri_1.0",
    "fri_2.0",
    "fri_5.0",
    "mas_-2.5",
    "mas_0.5",
    "mas_2.5",
    "mas_5.0",
    "mas_10.0",
]


def plot_adaptivity(load_run):

    # data loaded from load_run.pt file
    data = torch.load(load_run)
    print(data)
    data = data.cpu().numpy()
    all_round_errors = data
    for i in range(len(labels)):
        # all_round_errors[i] = np.sum(data[5 * i : 5 * i + 5]) / 5
        print(f"test {labels[i]} tracking error: {all_round_errors[i]}")
    # plt.bar(labels, all_round_errors, alpha=0.5)
    radar(all_round_errors, labels, load_run)


def radar(values, labels, name):

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 使雷达图封闭
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    # 绘图
    ax.fill(angles, values, alpha=0.1)

    ax.plot(
        angles,
        values,
        linewidth=2,
        linestyle="solid",
        label=name.split("/")[3]
        # + "_" + name.split("/")[4]
        ,
    )

    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    r_ticks = np.linspace(0, 0.8, 11)  # 根据你的数据范围调整
    ax.set_rgrids(r_ticks)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, nargs="+")
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

    if args.dir is not None:
        for path, dirname, pt in os.walk(args.dir):
            for file in pt:
                if file.endswith(".pt"):
                    print(file)
                    plot_adaptivity(os.path.join(path, file))
    if args.path is not None:
        for path in args.path:
            plot_adaptivity(path)
    plt.legend(loc="lower left")
    plt.show()
    plt.title("Tracking errors of different adaptivity strategies: History Designs")
    plt.savefig("radar_adaptivity.png")

# load .pt
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import torch
import os

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
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

re_index = [1, 0, 8, 4, 9, 2, 3, 5, 10, 7, 6]
scores = np.zeros((25, 11))


def plot_adaptivity(load_run, cnt):

    # data loaded from load_run.pt file
    data = torch.load(load_run)
    print(load_run)
    data = data.cpu().numpy()
    all_round_errors = data
    for i in range(len(labels)):
        # all_round_errors[i] = np.sum(data[5 * i : 5 * i + 5]) / 5
        print(f"test {labels[i]} tracking error: {all_round_errors[i]}")
        scores[i][re_index[cnt]] = all_round_errors[i]
    # plt.bar(labels, all_round_errors, alpha=0.5)
    radar(all_round_errors, labels, load_run)
    cnt += 1


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
        linestyle="dashed" if "teacher" in name else "solid",
        label=name.split("/")[3]
        # + "_" + name.split("/")[4]
        ,
    )

    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    r_ticks = np.linspace(0, 0.5, 11)  # 根据你的数据范围调整
    ax.set_rgrids(r_ticks)


def print_latex(scores):
    for i in range(25):
        print(f"{labels[i].split('_')[-1]}", end=" ")
        for j in range(11):
            print(f"& {scores[i][j]:.4f}", end=" ")
        print(r"\\")


if __name__ == "__main__":
    cnt = 0
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, nargs="+")
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    if args.dir is not None:
        for path, dirname, pt in os.walk(args.dir):
            for file in pt:
                if file.endswith(".pt"):
                    plot_adaptivity(os.path.join(path, file), cnt)
                    cnt += 1
    if args.path is not None:
        for path in args.path:
            plot_adaptivity(path, cnt)
            cnt += 1
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16)
    plt.show()
    plt.title("Linear Velocity Tracking Errors of Different Adaptive Strategies")
    plt.savefig("radar_adaptivity.jpg")
    # print_latex(scores)

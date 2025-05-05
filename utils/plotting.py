# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_with_rewards(rewards, accuracies, precisions, recalls, f1s):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    metrics = {
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1 Score": f1s,
    }
    reward_label = "Episode Reward"
    metric_names = list(metrics.keys())
    ma_window = 10
    ma_style = "-"
    metric_style = reward_style = "--"
    alpha_ma = 1
    alpha_reward = alpha_metric = 0.3
    reward_color = "tab:blue"
    metric_color = "tab:green"
    reward_color_ma = "navy"
    metric_color_ma = "darkgreen"

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    for idx, ax in enumerate(axs.flat):
        metric_name = metric_names[idx]
        metric_values = metrics[metric_name]
        ax2 = ax.twinx()

        lines = []

        (line1,) = ax.plot(
            rewards,
            label=reward_label,
            color=reward_color,
            alpha=alpha_reward,
            linestyle=reward_style,
        )
        lines.append(line1)

        if len(rewards) >= ma_window:
            ma_rewards = moving_average(rewards, ma_window)
            x_vals_rewards_ma = range(ma_window - 1, len(rewards))
            (line2,) = ax.plot(
                x_vals_rewards_ma,
                ma_rewards,
                linestyle=ma_style,
                label=f"{reward_label} (MA {ma_window})",
                color=reward_color_ma,
                linewidth=2,
            )
            lines.append(line2)

        (line3,) = ax2.plot(
            metric_values,
            label=metric_name,
            color=metric_color,
            alpha=alpha_metric,
            linestyle=metric_style,
        )
        lines.append(line3)

        if len(metric_values) >= ma_window:
            ma_metric = moving_average(metric_values, ma_window)
            x_vals_metric_ma = range(ma_window - 1, len(metric_values))
            (line4,) = ax2.plot(
                x_vals_metric_ma,
                ma_metric,
                linestyle=ma_style,
                color=metric_color_ma,
                linewidth=2,
                label=f"{metric_name} (MA {ma_window})",
            )
            lines.append(line4)

        ax.set_ylabel(reward_label)
        ax.tick_params(axis="y")
        ax2.set_ylabel(metric_name)
        ax2.tick_params(axis="y")
        ax.set_xlabel("Episode")
        ax.set_title(f"Reward vs {metric_name}")

        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="lower right")

    fig.suptitle("Training Performance Metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    return fig

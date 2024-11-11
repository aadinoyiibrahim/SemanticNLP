"""
This module contains functions to plot data.
"""

import matplotlib.pyplot as plt


def plot_value_counts(value_counts, title):
    """
    Plot the value counts of a categorical variable
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index, value_counts.values, color="blue")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yscale("log")

    total_count = value_counts.sum()
    for bar in bars:
        yval = bar.get_height()
        percentage = (yval / total_count) * 100
        label = f"{int(yval)} ({percentage:.1f}%)"
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval, label, ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.show()

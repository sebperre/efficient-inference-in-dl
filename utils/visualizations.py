from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

HOME_PATH = "/home/sebperre/programming-projects/efficient-inference-in-dl/logs"

def grouped_bar_chart(attributes, title, palette, file_name):
    correct_form = []

    for key, value in attributes.items():
        correct_form.append((key, *value.values()))

    attribute_names = list(list(attributes.values())[0].keys())
    
    data = pd.DataFrame(correct_form, columns=["Model", *(attribute_names)])
    data_melted = data.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data_melted, x="Model", y="Score", hue="Metric", palette=palette, ax=ax)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    fig.savefig(f"{file_name}.png", bbox_inches="tight")
    plt.close(fig)


# We will receive a dictionary with {Model Name: Time}
def plot_time(times, title, palette, file_name):
    adjusted_times, time_type = determine_time_type(times)

    df = pd.DataFrame(list(adjusted_times.items()), columns=["Model", "Time"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Time", palette=palette, ax=ax)

    ax.set_xlabel("Model")
    ax.set_ylabel(f"Time ({time_type})")
    ax.set_title(title)

    plt.tight_layout()

    fig.savefig(f"{file_name}.png", bbox_inches="tight")
    plt.close(fig)

def determine_time_type(times):
    # 5 minutes, 120 minutes
    for i, threshold in enumerate([300, 7200]):
        for time in times.values():
            if time <= threshold:
                if i == 0:
                    return times, "Seconds"
                else:
                    return {k: v / 60 for k, v in times.items()}, "Minutes"
    return {k: v / 3600 for k, v in times.items()}, "Hours"

def plot_heat_map(data, title, file_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.0f', cmap='coolwarm')
    plt.gca().invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{file_name}.png", bbox_inches="tight")
    plt.close(fig)

def plot_loss_per_epoch(num_epochs, train_losses, title, file_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{file_name}.png")

if __name__ == "__main__":
    # example = {"1": {"Accuracy": 0.75, "Recall": 0.6, "Precision": 0.5, "F1-Score": 0.8}, "2": {"Accuracy": 0.1, "Recall": 0.1, "Precision": 0.1, "F1-Score": 0.1}, "3": {"Accuracy": 0.1, "Recall": 0.1, "Precision": 0.1, "F1-Score": 0.1}, "4": {"Accuracy": 0.1, "Recall": 0.1, "Precision": 0.1, "F1-Score": 0.1}, "5": {"Accuracy": 0.1, "Recall": 0.1, "Precision": 0.1, "F1-Score": 0.1}, "Combined": {"Accuracy": 0.75, "Recall": 0.6, "Precision": 0.5, "F1-Score": 0.8}}
    # grouped_bar_chart(example, "Model Performance")
    example2 = {"1": 10000, "2": 8000, "3": 7300}
    plot_time(example2, "hi", "crest")
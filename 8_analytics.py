import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file1 = "test_data_marked_predictions_output.csv"
file2 = "test_without_smile_marked_predictions_output.csv"

df_with_emojis = pd.read_csv(file1)
df_without_emojis = pd.read_csv(file2)

category_order = ["positive", "neutral", "negative"]

count1 = df_with_emojis["predicted_label"].value_counts()
count2 = df_without_emojis["predicted_label"].value_counts()

count1 = count1[category_order].fillna(0)
count2 = count2[category_order].fillna(0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
count1.plot(kind="bar", color="skyblue")
plt.title("Распределение по классам для постов с эмоциями")
plt.xlabel("Эмоциональная окраска")
plt.ylabel("Количество постов")
plt.xticks(
    range(len(category_order)), category_order, rotation=0
)

plt.subplot(1, 2, 2)
count2.plot(kind="bar", color="salmon")
plt.title("Распределение по классам для постов без эмоций")
plt.xlabel("Эмоциональная окраска")
plt.ylabel("Количество постов")
plt.xticks(
    range(len(category_order)), category_order, rotation=0
)

plt.tight_layout()
plt.show()

count1 = (
    df_with_emojis["predicted_label"].value_counts(normalize=True) * 100
)
count2 = (
    df_without_emojis["predicted_label"].value_counts(normalize=True) * 100
)

count1 = count1[category_order].fillna(0)
count2 = count2[category_order].fillna(0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
count1.plot(kind="bar", color="skyblue")
plt.title("Распределение по классам для постов с эмоциями")
plt.xlabel("Эмоциональная окраска")
plt.ylabel("Количество постов в %")
plt.xticks(
    range(len(category_order)), category_order, rotation=0
)

plt.subplot(1, 2, 2)
count2.plot(kind="bar", color="salmon")
plt.title("Распределение по классам для постов без эмоций")
plt.xlabel("Эмоциональная окраска")
plt.ylabel("Количество постов в %")
plt.xticks(
    range(len(category_order)), category_order, rotation=0
)

plt.tight_layout()
plt.show()

df_with_emojis["probabilities"] = df_with_emojis["probabilities"].apply(
    lambda x: np.array(eval(x))
)
df_without_emojis["probabilities"] = df_without_emojis["probabilities"].apply(
    lambda x: np.array(eval(x))
)

probabilities_array1 = np.array(df_with_emojis["probabilities"].to_list())
probabilities_array2 = np.array(df_without_emojis["probabilities"].to_list())

mean_probabilities1 = probabilities_array1.mean(axis=0)
mean_probabilities2 = probabilities_array2.mean(axis=0)

class_labels = ["positive", "neutral", "negative"]

x = np.arange(len(class_labels))

width = 0.35

fig, ax = plt.subplots()

ax.bar(
    x - width / 2, mean_probabilities1, width, label="посты с эмоциями", color="#4CAF50"
)
ax.bar(
    x + width / 2, mean_probabilities2, width, label="посты без эмоций", color="#2196F3"
)

ax.set_xlabel("Эмоциональная окраска")
ax.set_ylabel("Средняя вероятность")
ax.set_title("Средняя вероятность для каждого класса")
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()

plt.show()


df_with_emojis["positive_prob"] = df_with_emojis["probabilities"].apply(lambda x: x[0])
df_with_emojis["neutral_prob"] = df_with_emojis["probabilities"].apply(lambda x: x[1])
df_with_emojis["negative_prob"] = df_with_emojis["probabilities"].apply(lambda x: x[2])

df_without_emojis["positive_prob"] = df_without_emojis["probabilities"].apply(
    lambda x: x[0]
)
df_without_emojis["neutral_prob"] = df_without_emojis["probabilities"].apply(
    lambda x: x[1]
)
df_without_emojis["negative_prob"] = df_without_emojis["probabilities"].apply(
    lambda x: x[2]
)

fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

sns.kdeplot(df_with_emojis["positive_prob"], ax=axes[0], fill=True, label="Со смайликами")
sns.kdeplot(df_without_emojis["positive_prob"], ax=axes[0], fill=True, label="Без смайликов")
axes[0].set_title("Плотность распределения для позитивных вероятностей")
axes[0].set_xlabel("Вероятность")
axes[0].set_ylabel("Плотность")
axes[0].legend()

sns.kdeplot(df_with_emojis["neutral_prob"], ax=axes[1], fill=True, label="Со смайликами")
sns.kdeplot(df_without_emojis["neutral_prob"], ax=axes[1], fill=True, label="Без смайликов")
axes[1].set_title("Плотность распределения для нейтральных вероятностей")
axes[1].set_xlabel("Вероятность")
axes[1].set_ylabel("Плотность")
axes[1].legend()

sns.kdeplot(df_with_emojis["negative_prob"], ax=axes[2], fill=True, label="Со смайликами")
sns.kdeplot(df_without_emojis["negative_prob"], ax=axes[2], fill=True, label="Без смайликов")
axes[2].set_title("Плотность распределения для негативных вероятностей")
axes[2].set_xlabel("Вероятность")
axes[2].set_ylabel("Плотность")
axes[2].legend()

plt.tight_layout(pad=3.0)
plt.legend(loc="upper right")
plt.show()

plt.boxplot(
    [df_with_emojis["positive_prob"], df_without_emojis["positive_prob"]],
    labels=["Со смайликами", "Без смайликов"],
)
plt.title("Позитивные вероятности с и без смайликов")
plt.ylabel("Позитивная вероятность")
plt.show()

accuracy_with_emojis = (
    df_with_emojis["label"] == df_with_emojis["predicted_label"]
).mean()
accuracy_without_emojis = (
    df_without_emojis["label"] == df_without_emojis["predicted_label"]
).mean()

print(f"Accuracy with emojis: {accuracy_with_emojis}")
print(f"Accuracy without emojis: {accuracy_without_emojis}")

from scipy.stats import shapiro

stat_with, p_with = shapiro(df_with_emojis["positive_prob"])
stat_without, p_without = shapiro(df_without_emojis["positive_prob"])

print(f"P-value для данных со смайликами: {p_with}")
print(f"P-value для данных без смайликов: {p_without}")

if p_with < 0.05 or p_without < 0.05:
    print("Данные распределены ненормально.")
else:
    print("Данные распределены нормально.")

from scipy.stats import wilcoxon

stat, p = wilcoxon(df_with_emojis["positive_prob"], df_without_emojis["positive_prob"])
print(f"Wilcoxon-statistic: {stat}, P-value: {p}")

from scipy.stats import ranksums

stat, p = ranksums(df_with_emojis["positive_prob"], df_without_emojis["positive_prob"])
print(f"Ranksums Test: Statistic={stat}, p-value={p}")

median_with = df_with_emojis["positive_prob"].median()
median_without = df_without_emojis["positive_prob"].median()
print(f"Median Positive Probability - With Emojis: {median_with}")
print(f"Median Positive Probability - Without Emojis: {median_without}")

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axes[0].hist(df_with_emojis["positive_prob"], bins=30, color="green", alpha=0.7)
axes[0].set_title("Распределение позитивных вероятностей - Со смайликами")
axes[0].set_xlabel("Вероятность")
axes[0].set_ylabel("Частота")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

axes[1].hist(df_without_emojis["positive_prob"], bins=30, color="blue", alpha=0.7)
axes[1].set_title("Распределение позитивных вероятностей - Без смайликов")
axes[1].set_xlabel("Вероятность")
axes[1].set_ylabel("Частота")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

print(df_with_emojis.head(5))
print(df_without_emojis.head(5))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(true_labels, predicted_labels):
    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Precision": precision_score(true_labels, predicted_labels, average="weighted"),
        "Recall": recall_score(true_labels, predicted_labels, average="weighted"),
        "F1 Score": f1_score(true_labels, predicted_labels, average="weighted"),
    }


metrics_with_emojis = compute_metrics(
    df_with_emojis["label"], df_with_emojis["predicted_label"]
)

metrics_without_emojis = compute_metrics(
    df_without_emojis["label"], df_without_emojis["predicted_label"]
)

metrics_table = pd.DataFrame(
    {"With Emojis": metrics_with_emojis, "Without Emojis": metrics_without_emojis}
)

print(metrics_table)

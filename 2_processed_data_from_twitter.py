import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------
# Шаг 1: Загрузка и объединение данных из двух файлов
# ---------------------------------------------------
df1 = pd.read_csv('rusentitweet_train.csv')

df2 = pd.read_csv('rusentitweet_test.csv')

combined_df = pd.concat([df1, df2], ignore_index=True)

combined_file_path = os.path.join(output_dir, 'train_dataset_sentiment.csv')
combined_df.to_csv(combined_file_path, index=False)

print(f"Файлы успешно объединены в '{combined_file_path}'")

# ---------------------------------------------------
# Шаг 2: Сбалансировка данных (равное количество классов)
# ---------------------------------------------------
target_size = 2200

balanced_df = pd.concat([
    combined_df[combined_df['label'] == label].sample(n=target_size, replace=True, random_state=42)
    for label in ['positive', 'neutral', 'negative']
])

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_file_path = os.path.join(output_dir, 'balanced_data.csv')
balanced_df.to_csv(balanced_file_path, index=False)

print(f"Сбалансированные данные сохранены в '{balanced_file_path}'")

# ---------------------------------------------------
# Шаг 3: Фильтрация данных и удаление столбца 'id'
# ---------------------------------------------------
filtered_df = balanced_df[balanced_df['label'].isin(['positive', 'neutral', 'negative'])].drop(columns=['id'])

filtered_file_path = os.path.join(output_dir, 'balanced_data_final.csv')
filtered_df.to_csv(filtered_file_path, index=False)

print(f"Фильтрация завершена. Данные сохранены в '{filtered_file_path}'")

# ---------------------------------------------------
# Шаг 4: Построение гистограммы распределения классов
# ---------------------------------------------------
# Подсчет количества записей для каждого класса
label_counts = filtered_df['label'].value_counts()

plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', rot=0)
plt.title('Distribution of Labels', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Сохранение графика в новую папку
histogram_path = os.path.join(output_dir, 'label_distribution.png')
plt.savefig(histogram_path)
plt.show()

print(f"Гистограмма распределения классов сохранена в '{histogram_path}'")

import json
import matplotlib.pyplot as plt

with open('classification_head_experiments/results.json', 'r') as file:
    data = json.load(file)

heads = data.keys()
steps = range(1, len(next(iter(data.values()))) + 1)

metrics = ['accuracy', 'f1', 'precision', 'recall']

metric_labels = {
    'accuracy': 'Точность (Accuracy)',
    'f1': 'F1-Мера',
    'precision': 'Точность (Precision)',
    'recall': 'Полнота (Recall)'
}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for head in heads:
        values = [entry[metric] for entry in data[head]]
        plt.plot(steps, values, marker='o', label=head)

    plt.title(f'{metric_labels[metric]} на каждом шаге для каждого классификатора')
    plt.xlabel('Шаг')
    plt.ylabel(metric_labels[metric])
    plt.legend(title="Классификаторы")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

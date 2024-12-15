import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score


def analyze_metrics(file_path):
    results = pd.read_csv(file_path)

    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    results['predicted_label_num'] = results['predicted_label'].map(label_mapping)

    results = results.dropna(subset=['label', 'predicted_label_num'])

    true_labels = results['label'].map(label_mapping)
    predicted_labels = results['predicted_label_num']

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    class_report = classification_report(true_labels, predicted_labels, target_names=['positive', 'neutral', 'negative'])

    print(f"Результаты для файла: {file_path}")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Средняя точность (Precision): {precision:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"F1-мера: {f1:.4f}")
    print("Подробный отчет о метриках:")
    print(class_report)

    return accuracy, precision, recall, f1, class_report


file_with_smile = 'test_data_marked_predictions_output.csv'
file_without_smile = 'test_without_smile_marked_predictions_output.csv'

metrics_with_smile = analyze_metrics(file_with_smile)
metrics_without_smile = analyze_metrics(file_without_smile)

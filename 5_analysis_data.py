import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

file_path = 'test_without_smile_marked_predictions_output.csv'
results = pd.read_csv(file_path)

label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
results['predicted_label_num'] = results['predicted_label'].map(label_mapping)

results = results.dropna(subset=['label', 'predicted_label_num'])

correct_predictions = (results['label'].map(label_mapping) == results['predicted_label_num']).sum()

accuracy = accuracy_score(results['label'].map(label_mapping), results['predicted_label_num'])

class_report = classification_report(results['label'].map(label_mapping), results['predicted_label_num'], target_names=['positive', 'neutral', 'negative'])

print(f"Количество верных предсказаний: {correct_predictions}")
print(f"Точность модели: {accuracy:.4f}")
print("Подробный отчет о метриках:")
print(class_report)









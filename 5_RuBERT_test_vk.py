import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_save_path = "fine_tuned_rubert_lr6.443945724388442e-05_batch16"

tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_prediction(test_texts, model, tokenizer):
    max_seq_len = 512

    print(f"Максимальная длина последовательности для тестовых данных: {max_seq_len}")

    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1).cpu().numpy()

    return predictions, probs.cpu().numpy()


file_path = 'vk_test_marked/test_without_smile_marked.csv'
data = pd.read_csv(file_path)

predicted_labels, probabilities = get_prediction(data['text'].tolist(), model, tokenizer)

label_mapping = {0: 'positive', 1: 'neutral', 2: 'negative'}
predicted_labels_text = [label_mapping[label] for label in predicted_labels]

results = data.copy()
results['predicted_label'] = predicted_labels_text
results['probabilities'] = [list(prob) for prob in probabilities]

output_file_path = 'test_without_smile_marked_predictions_output.csv'
results.to_csv(output_file_path, index=False)

print(results.head())



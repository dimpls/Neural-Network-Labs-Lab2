import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
import os
import json


class ClassificationHeadA(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadA, self).__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, num_labels)

    def forward(self, features):
        x = self.dense(features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHeadB(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadB, self).__init__()
        self.dense1 = nn.Linear(input_dim, 128)
        self.dense2 = nn.Linear(128, 64)
        self.out_proj = nn.Linear(64, num_labels)

    def forward(self, features):
        x = torch.relu(self.dense1(features))
        x = torch.relu(self.dense2(x))
        x = self.out_proj(x)
        return x


class ClassificationHeadC(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadC, self).__init__()
        self.out_proj = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        return self.out_proj(features)


class ClassificationHeadD(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadD, self).__init__()
        self.dense1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512, 128)
        self.out_proj = nn.Linear(128, num_labels)

    def forward(self, features):
        x = torch.relu(self.dense1(features))
        x = self.dropout(x)
        x = torch.relu(self.dense2(x))
        x = self.out_proj(x)
        return x


class ClassificationHeadE(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadE, self).__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.norm = nn.BatchNorm1d(256)
        self.out_proj = nn.Linear(256, num_labels)

    def forward(self, features):
        x = torch.relu(self.dense(features))
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class ClassificationHeadF(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ClassificationHeadF, self).__init__()
        self.dense1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.1)
        self.out_proj = nn.Linear(64, num_labels)

    def forward(self, features):
        x = torch.relu(self.dense1(features))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, classification_head):
        super(CustomModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.classifier = classification_head

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # ---------------------------------------------
    # Шаг 1: Загрузка и подготовка данных
    # ---------------------------------------------
    file_path = "processed_data/dataset_for_train.csv"
    df = pd.read_csv(file_path)

    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['label'] = df['label'].map(label_mapping)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    seq_len_train = [len(str(i).split()) for i in train_texts]
    seq_len_test = [len(str(i).split()) for i in test_texts]
    max_seq_len = max(max(seq_len_test), max(seq_len_train))

    print(f"Максимальная длина последовательности: {max_seq_len}")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_seq_len)

    train_data = HFDataset.from_dict({"text": train_texts, "label": train_labels})
    test_data = HFDataset.from_dict({"text": test_texts, "label": test_labels})

    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    train_data = train_data.remove_columns(["text"]).with_format("torch")
    test_data = test_data.remove_columns(["text"]).with_format("torch")

    # ---------------------------------------------
    # Шаг 2: Настройка метрик
    # ---------------------------------------------
    def compute_metrics(predictions, labels):
        predictions = torch.argmax(predictions, dim=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # ---------------------------------------------
    # Шаг 3: Эксперименты с разными слоями классификации
    # ---------------------------------------------
    classification_heads = {
        "HeadA": ClassificationHeadA,
        "HeadB": ClassificationHeadB,
        "HeadC": ClassificationHeadC,
        "HeadD": ClassificationHeadD,
        "HeadE": ClassificationHeadE,
        "HeadF": ClassificationHeadF
    }

    results = {}
    for head_name, HeadClass in classification_heads.items():
        print(f"\nTraining with {head_name}")

        model = CustomModel('DeepPavlov/rubert-base-cased', num_labels=3, classification_head=HeadClass(768, 3)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=16)

        epoch_metrics = []

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                loss, logits = model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"{head_name} - Epoch {epoch + 1}, Training Loss: {total_loss / len(train_dataloader)}")

            model.eval()
            all_predictions, all_labels = [], []
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    output = model(input_ids, attention_mask)
                    logits = output if isinstance(output, torch.Tensor) else output[1]

                    all_predictions.append(logits)
                    all_labels.append(labels)

            all_predictions = torch.cat(all_predictions).cpu()
            all_labels = torch.cat(all_labels).cpu()
            metrics = compute_metrics(all_predictions, all_labels)
            epoch_metrics.append(metrics)

            print(f"{head_name} - Epoch {epoch + 1}, Metrics: {metrics}")

        results[head_name] = epoch_metrics

    # ---------------------------------------------
    # Сохранение результатов
    # ---------------------------------------------
    save_path = "classification_head_experiments"
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Experiment results saved to {save_path}/results.json")


if __name__ == "__main__":
    main()

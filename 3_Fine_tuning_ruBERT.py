import optuna
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
import os
import json


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # ---------------------------------------------
    # Шаг 1: Загрузка и подготовка данных
    # ---------------------------------------------
    file_path = "processed_data/train_dataset_final.csv"
    df = pd.read_csv(file_path)

    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['label'] = df['label'].map(label_mapping)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

    train_data = HFDataset.from_dict({"text": train_texts, "label": train_labels})
    test_data = HFDataset.from_dict({"text": test_texts, "label": test_labels})

    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    train_data = train_data.remove_columns(["text"]).with_format("torch")
    test_data = test_data.remove_columns(["text"]).with_format("torch")

    # ---------------------------------------------
    # Шаг 2: Настройка метрик
    # ---------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)

        class_report = classification_report(labels, predictions, target_names=['positive', 'neutral', 'negative'])
        print("Classification Report:\n", class_report)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # ---------------------------------------------
    # Шаг 3: Настройка модели для подбора гиперпараметров
    # ---------------------------------------------
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            'DeepPavlov/rubert-base-cased',
            num_labels=3
        )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        num_train_epochs=5,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )

    # ---------------------------------------------
    # Шаг 4: Подбор гиперпараметров с помощью Optuna
    # ---------------------------------------------
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
            "max_seq_length": trial.suggest_categorical("max_seq_length", [256, 512])
        }

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=3  
    )

    print("Best hyperparameters:", best_run.hyperparameters)

    # ---------------------------------------------
    # Шаг 5: Сохранение модели, токенизатора и параметров
    # ---------------------------------------------
    model_save_path = f"fine_tuned_rubert_lr{best_run.hyperparameters['learning_rate']}_batch{best_run.hyperparameters['per_device_train_batch_size']}"
    os.makedirs(model_save_path, exist_ok=True)

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    with open(os.path.join(model_save_path, "best_hyperparameters.json"), "w") as f:
        json.dump(best_run.hyperparameters, f, indent=4)

    print(f"Модель и токенизатор сохранены в папке: {model_save_path}")


if __name__ == "__main__":
    main()

import json
import numpy as np
import librosa
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class DeceptionTrainer:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "truth", 1: "deception"},
            label2id={"truth": 0, "deception": 1},
        )

    def load_dataset(self, json_path):
        with open(json_path, "r") as f:
            payload = json.load(f)
        return Dataset.from_dict({
            "audio": [item["audio"]["path"] for item in payload["data"]],
            "label": [item["label"] for item in payload["data"]],
        })

    def preprocess(self, examples):
        audio_arrays = []
        for path in examples["audio"]:
            audio, _ = librosa.load(path, sr=16000)
            audio_arrays.append(audio)
        inputs = self.processor(
            audio_arrays,
            sampling_rate=16000,
            max_length=int(16000 * 30),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = examples["label"]
        return inputs

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, dataset_path, output_dir="./deception-model"):
        dataset = self.load_dataset(dataset_path)
        split = dataset.train_test_split(test_size=0.2)
        processed = split.map(
            self.preprocess,
            batched=True,
            batch_size=4,
            remove_columns=split["train"].column_names,
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=8,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=200,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=processed["train"],
            eval_dataset=processed["test"],
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        return trainer.evaluate()

if __name__ == "__main__":
    trainer = DeceptionTrainer()
    results = trainer.train("dataset.json")
    print("Training Results:", results)

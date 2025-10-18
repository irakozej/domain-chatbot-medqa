import tensorflow as tf
from transformers import (
    T5TokenizerFast, TFT5ForConditionalGeneration,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
from pathlib import Path

def load_dataset_splits():
    """Load processed CSV files."""
    base = Path("data/processed")
    train = pd.read_csv(base / "train.csv")
    val = pd.read_csv(base / "val.csv")
    return train, val

def tokenize_function(examples, tokenizer, max_input=256, max_target=128):
    inputs = ["question: " + q.strip() for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding="max_length")
    labels = tokenizer(examples["answer"], max_length=max_target, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    model_name = "t5-small"
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = TFT5ForConditionalGeneration.from_pretrained(model_name)

    train_df, val_df = load_dataset_splits()
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tokenized_train = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    tf_train = tokenized_train.to_tf_dataset(columns=["input_ids", "attention_mask", "labels"], shuffle=True, batch_size=8, collate_fn=data_collator)
    tf_val = tokenized_val.to_tf_dataset(columns=["input_ids", "attention_mask", "labels"], shuffle=False, batch_size=8, collate_fn=data_collator)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer)

    model.fit(tf_train, validation_data=tf_val, epochs=2)
    model.save_pretrained("models/t5_medqa_final")
    tokenizer.save_pretrained("models/t5_medqa_final")

    print("✅ Model saved to models/t5_medqa_final")

if __name__ == "__main__":
    train_model()

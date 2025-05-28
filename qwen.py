import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import pandas as pd
import torch
import mlflow  # Ensure mlflow is installed and configured if you intend to use it.

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # Changed from AutoModelForCausalLM
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,  # Added for good practice
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="binary")
    return {"accuracy": accuracy, "f1": f1}


def load_and_prepare_data(filepath="assets/douban_movie.csv"):
    """
    Loads the dataset, handles missing values, and creates the target variable.
    A movie is considered 'loved' (1) if Star >= 3,
    otherwise 'not loved' (0).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Please check the path.")
        # Consider raising an exception or returning None consistently
        raise
        # return None

    # Select necessary columns and handle missing values
    df = df[["Comment", "Star"]].copy()
    df.dropna(subset=["Comment", "Star"], inplace=True)
    df["Star"] = pd.to_numeric(df["Star"], errors="coerce")
    df.dropna(subset=["Star"], inplace=True)

    # Define 'Sentiment': 1 if Star >= 3 (Loved), 0 otherwise (Not Loved)
    df["Sentiment"] = (
        df["Star"].apply(lambda x: 1 if x >= 3 else 0).astype(int)
    )  # Ensure integer type

    print(f"Loaded {len(df)} reviews.")
    print(f"Class distribution:\n{df['Sentiment'].value_counts(normalize=True)}")

    return df[["Comment", "Sentiment"]]


csv_file_path = "assets/douban_movie.csv"  # Make sure this path is correct

if __name__ == "__main__":
    try:
        df = load_and_prepare_data(csv_file_path)
    except FileNotFoundError:
        # If load_and_prepare_data raises an error, you might want to exit or handle it
        print(f"Failed to load data from {csv_file_path}. Exiting.")
        exit()

    if df is None or df.empty:
        print("No data loaded. Exiting.")
        exit()

    dataset = Dataset.from_pandas(df)

    try:
        train_test_split = dataset.train_test_split(
            test_size=0.2, stratify_by_column="Sentiment"
        )
    except Exception as e:
        print(f"Could not stratify, falling back to regular split: {e}")
        train_test_split = dataset.train_test_split(test_size=0.2)

    dataset_dict = DatasetDict(
        {"train": train_test_split["train"], "validation": train_test_split["test"]}
    )

    model_name = "Qwen/Qwen3-8B"
    print(f"Using model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["Comment"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    tokenized_train_dataset = dataset_dict["train"].map(tokenize_function, batched=True)
    tokenized_validation_dataset = dataset_dict["validation"].map(
        tokenize_function, batched=True
    )

    # Remove original text column and rename 'Sentiment' to 'labels'
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(
        ["Comment", "Star"]
        if "Star" in tokenized_train_dataset.column_names
        else ["Comment"]
    )
    tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(
        ["Comment", "Star"]
        if "Star" in tokenized_validation_dataset.column_names
        else ["Comment"]
    )

    tokenized_train_dataset = tokenized_train_dataset.rename_column(
        "Sentiment", "labels"
    )
    tokenized_validation_dataset = tokenized_validation_dataset.rename_column(
        "Sentiment", "labels"
    )

    tokenized_train_dataset.set_format("torch")
    tokenized_validation_dataset.set_format("torch")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Number of unique labels
    num_labels = df["Sentiment"].nunique()
    print(f"Number of labels: {num_labels}")

    id2label = {0: "NOT_LOVED", 1: "LOVED"}
    label2id = {"NOT_LOVED": 0, "LOVED": 1}

    # Load base model for Sequence Classification with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        num_labels=num_labels,  # Specify the number of labels
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        trust_remote_code=True,
        # config=AutoConfig.from_pretrained(model_name, pad_token_id=tokenizer.pad_token_id, num_labels=num_labels, trust_remote_code=True)
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Common attention projections
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP layers in Qwen
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./qwen_sentiment_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=50,
        label_names=["labels"],
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="mlflow"
        if "MLFLOW_TRACKING_URI" in os.environ
        else "none",  # Conditional MLflow
        # save_total_limit=2, # Save only the last 2 checkpoints
        # load_best_model_at_end=True, # Load the best model at the end of training
        metric_for_best_model="accuracy",
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,  # Pass tokenizer for proper padding and handling
        data_collator=data_collator,  # Use the data collator
        compute_metrics=compute_metrics,  # Add if you define metrics
    )

    if training_args.report_to == "mlflow":
        mlflow.start_run()
        # Log hyperparameters
        mlflow.log_params(training_args.to_dict())
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("lora_r", lora_config.r)
        mlflow.log_param("lora_alpha", lora_config.lora_alpha)

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the fine-tuned model (adapters)
    final_model_path = "./qwen_sentiment_finetuned/final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model and tokenizer saved to {final_model_path}")

    # End MLflow run
    if training_args.report_to == "mlflow":
        # Log final metrics if compute_metrics was used
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)
        mlflow.end_run()

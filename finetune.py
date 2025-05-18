import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from evaluate import load
import numpy as np
from utils import load_and_prepare_data
# import wandb
import mlflow
import datetime


model_name = "Qwen/Qwen3-8B"
model_id = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

QWEN_PAD_TOKEN_ID = 151643
# Ensure the tokenizer is configured to use Qwen's specified PAD token
# which is its BOS token.
if tokenizer.pad_token_id != QWEN_PAD_TOKEN_ID:
    print(f"Tokenizer's current pad_token_id is {tokenizer.pad_token_id}.")
    print(
        f"Setting tokenizer's pad_token_id to Qwen's specified BOS/PAD token ID: {QWEN_PAD_TOKEN_ID}"
    )
    tokenizer.pad_token_id = QWEN_PAD_TOKEN_ID

    # Also set the pad_token string. Since Qwen uses BOS as PAD:
    if tokenizer.bos_token_id == QWEN_PAD_TOKEN_ID:
        tokenizer.pad_token = tokenizer.bos_token
        print(f"Set tokenizer.pad_token to its BOS token: '{tokenizer.bos_token}'")
    else:
        # This case should ideally not occur if QWEN_PAD_TOKEN_ID is indeed the bos_token_id
        # as per the generation_config. Fallback to decode if needed.
        bos_token_for_pad = tokenizer.decode([QWEN_PAD_TOKEN_ID])
        tokenizer.pad_token = bos_token_for_pad
        print(
            f"Warning: QWEN_PAD_TOKEN_ID ({QWEN_PAD_TOKEN_ID}) does not match tokenizer.bos_token_id ({tokenizer.bos_token_id})."
        )
        print(
            f"Set tokenizer.pad_token by decoding {QWEN_PAD_TOKEN_ID} to: '{tokenizer.pad_token}'"
        )

# Verify tokenizer settings
print(
    f"Using Tokenizer - pad_token: '{tokenizer.pad_token}', pad_token_id: {tokenizer.pad_token_id}, bos_token: '{tokenizer.bos_token}', bos_token_id: {tokenizer.bos_token_id}"
)

# Load your CSV
csv_file_path = "douban_movie.csv"

df = load_and_prepare_data(csv_file_path)

dataset = Dataset.from_pandas(df)

train_test_split = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict(
    {"train": train_test_split["train"], "validation": train_test_split["test"]}
)

print(dataset_dict)


def tokenize_function(examples):
    # For sentiment analysis, you are classifying the input text.
    # Max length should be chosen based on your dataset and model capabilities.
    return tokenizer(
        examples["Comment"], truncation=True, padding="max_length", max_length=512
    )


tokenized_train_dataset = dataset_dict["train"].map(tokenize_function, batched=True)
tokenized_validation_dataset = dataset_dict["validation"].map(
    tokenize_function, batched=True
)

# Remove original text column as it's no longer needed after tokenization
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["Comment"])
tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(["Comment"])
# Rename the 'Sentiment' column to 'labels' as expected by the model
tokenized_train_dataset = tokenized_train_dataset.rename_column("Sentiment", "labels")
tokenized_validation_dataset = tokenized_validation_dataset.rename_column(
    "Sentiment", "labels"
)

tokenized_train_dataset.set_format("torch")
tokenized_validation_dataset.set_format("torch")


# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Optional
    bnb_4bit_quant_type="nf4",  # nf4 is often recommended
    bnb_4bit_compute_dtype=torch.bfloat16,  # Or torch.float16 depending on GPU
)

# Number of labels for sentiment analysis (e.g., 0 for negative, 1 for positive)
num_labels = 2  # Or len(df['label'].unique())

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    num_labels=num_labels,
    trust_remote_code=True,
    device_map="auto",  # Automatically distributes model layers across available GPUs/CPU
)

# It's also good practice to ensure the model's pad_token_id matches the tokenizer's
# if it was already set but different (though less common for this specific error).
if model.config.pad_token_id != tokenizer.pad_token_id:
    print(
        f"Warning: model.config.pad_token_id ({model.config.pad_token_id}) "
        f"differs from tokenizer.pad_token_id ({tokenizer.pad_token_id}). "
        f"Setting model.config.pad_token_id to match tokenizer."
    )
    model.config.pad_token_id = tokenizer.pad_token_id

# If the model has such a config, you might want to ensure it's set appropriately,
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False  # Often recommended for training
if hasattr(model.config, "enable_thinking"):  # Qwen3 specific
    # For sentiment analysis, a direct classification approach is usually desired.
    # Disabling thinking mode during fine-tuning for classification tasks might be beneficial.
    # model.config.enable_thinking = False # This would be for inference, check if it affects training
    pass  # Consult Qwen docs for fine-tuning regarding this.

classification_head_name = "score"

lora_target_modules = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    inference_mode=False,
    r=8,  # LoRA attention dimension (rank) - common values 8, 16, 32, 64
    lora_alpha=16,  # LoRA alpha
    lora_dropout=0.1,  # LoRA dropout
    target_modules=lora_target_modules,  # Modules to apply LoRA to.
    modules_to_save=[classification_head_name],
)

if model.config.pad_token_id != tokenizer.pad_token_id:
    print(f"Model's current config.pad_token_id is {model.config.pad_token_id}.")
    print(
        f"Setting model.config.pad_token_id to match tokenizer's pad_token_id: {tokenizer.pad_token_id}"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

# Verify model config
print(f"Model config - pad_token_id: {model.config.pad_token_id}")

# model.gradient_checkpointing_enable() # Can save memory but slows down training

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()  # See how many parameters are being trained

# Define metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    # For binary classification, specify pos_label for F1 if needed, or average.
    # Ensure labels are correctly formatted for the metric.
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]  # or "binary" if pos_label=1
    return {"accuracy": accuracy, "f1": f1}


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

output_dir = "./qwen_sentiment_finetuned"
per_device_train_batch_size = (
    2  # Adjust based on your VRAM (1, 2, 4, 8 are common for large models)
)
per_device_eval_batch_size = per_device_train_batch_size * 2
gradient_accumulation_steps = (
    4  # Effective batch size = batch_size * num_gpus * grad_accumulation
)
learning_rate = 2e-4  # Common for LoRA, can be 1e-4, 2e-4, 3e-4
num_train_epochs = 3  # Adjust as needed (1-5 is common for fine-tuning)
logging_steps = 10
save_steps = 50  # Save checkpoints periodically
eval_steps = 50  # Evaluate periodically

mlflow.set_experiment("Qwen3 PEFT")
date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=logging_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,  # Keep only the best and the last checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # or "f1"
    report_to="mlflow",
    run_name=f"Qweb-8B-QLoRA-{date_time}",
    fp16=False,  # Set to True if your GPU supports FP16 and you are not using bfloat16 compute_dtype in BnB
    bf16=True
    if torch.cuda.is_bf16_supported()
    and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16
    else False,  # Set to True if using bfloat16
    remove_unused_columns=False,  # Important for PEFT
    optim="paged_adamw_8bit",  # Can save memory, requires bitsandbytes
    label_names=["labels"],
)

trainer = Trainer(
    model=peft_model,  # Use the PEFT model
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the LoRA adapters
peft_model.save_pretrained(f"{output_dir}/final_lora_adapters")
tokenizer.save_pretrained(f"{output_dir}/final_lora_adapters")  # Save tokenizer as well

# To save the full model (if needed, larger):
# peft_model.save_pretrained(f"{output_dir}/final_full_model", safe_serialization=True) # This might save the merged model or adapters depending on PEFT version.
# For merging and saving the full model:
# merged_model = peft_model.merge_and_unload()
# merged_model.save_pretrained(f"{output_dir}/final_merged_model")
# tokenizer.save_pretrained(f"{output_dir}/final_merged_model")

print(f"Fine-tuned LoRA adapters saved to {output_dir}/final_lora_adapters")

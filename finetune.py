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
from comment_classification import load_and_prepare_data
import wandb
model_name = "Qwen/Qwen3-30B-A3B"
model_id = "Qwen/Qwen3-30B-A3B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set padding token if not already set (Qwen models usually handle this)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# For Qwen's "thinking mode" - for classification, you likely don't need complex reasoning.
# If the model has such a config, you might want to ensure it's set appropriately,
# though for fine-tuning a classification head, this might be less critical than for generation.
# Check model.config for relevant attributes.
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False  # Often recommended for training
if hasattr(model.config, "enable_thinking"):  # Qwen3 specific
    # For sentiment analysis, a direct classification approach is usually desired.
    # Disabling thinking mode during fine-tuning for classification tasks might be beneficial.
    # model.config.enable_thinking = False # This would be for inference, check if it affects training
    pass  # Consult Qwen docs for fine-tuning regarding this.


# It's crucial to identify the target modules for LoRA correctly for your specific Qwen MoE model.
# Common targets are 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'.
# For MoE models, you might need to be careful about which experts' layers to target,
# or if there are shared layers. Unsloth handles this well for supported models.
# If using transformers directly, you may need to inspect the model architecture.
# A general approach is to target all linear layers in the attention and MLP blocks.
# `peft` can sometimes infer this, or you might need to list them.


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
    # If unsure, you can try omitting this and let PEFT attempt to find suitable layers,
    # or consult Qwen fine-tuning guides for specific recommendations.
    # For some models, it might be e.g. ["Wqkv", "out_proj", "w1", "w2"]
    # Or just ["all-linear"] if supported and desired.
)

# Ensure the model's classification head is trainable if not already
for param in model.parameters():
    if (
        hasattr(model, "score") and param in model.score.parameters()
    ):  # 'score' is often the name of the classification head
        param.requires_grad = True
    elif (
        hasattr(model, "classifier") and param in model.classifier.parameters()
    ):  # Another common name
        param.requires_grad = True


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
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,  # Keep only the best and the last checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # or "f1"
    report_to="wandb",
    fp16=False,  # Set to True if your GPU supports FP16 and you are not using bfloat16 compute_dtype in BnB
    bf16=True
    if torch.cuda.is_bf16_supported()
    and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16
    else False,  # Set to True if using bfloat16
    remove_unused_columns=False,  # Important for PEFT
    # optim="paged_adamw_8bit" # Can save memory, requires bitsandbytes
    # optim="adamw_torch_fused" # If available and using PyTorch >= 2.0
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

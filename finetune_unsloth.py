import pandas as pd
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import mlflow  # Ensure mlflow is installed and configured if you intend to use it.


# Configuration
CSV_FILE_PATH = "assets/douban_movie.csv"  # Path to your CSV file
MODEL_NAME = "Qwen/Qwen3-8B" # Base Qwen model. Unsloth will handle 4-bit quantization.
                                      # You can replace this with a specific Unsloth 4-bit model if preferred e.g. "unsloth/qwen2-7b-instruct-bnb-4bit"
OUTPUT_DIR = "./qwen3_sentiment_finetuned" # Directory to save finetuned model and logs
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_lora_adapter")

# Model and Training Parameters
MAX_SEQ_LENGTH = 1024  # Adjust based on VRAM and average comment length. V100s should handle this.
LOAD_IN_4BIT = True
LORA_R = 16            # LoRA rank
LORA_ALPHA = 32        # LoRA alpha
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Batch size and gradient accumulation for 8x V100s (16GB or 32GB)
# Effective batch size = PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
# For V100 16GB, PER_DEVICE_TRAIN_BATCH_SIZE might be 1 or 2.
# For V100 32GB, PER_DEVICE_TRAIN_BATCH_SIZE could be 2, 4, or higher.
# Adjust these based on your specific V100 VRAM and observed memory usage.
# Example for V100 16GB aiming for effective batch size of 64:
# PER_DEVICE_TRAIN_BATCH_SIZE = 1, NUM_GPUS = 8, GRADIENT_ACCUMULATION_STEPS = 8 => 1*8*8 = 64
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_STEPS = 20
OPTIMIZER = "adamw_8bit" # Or "adamw_bnb_8bit" if using bitsandbytes optimizer

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the sentiment of the following movie review as positive or negative.

### Input:
{}

### Response:
{}"""
EOS_TOKEN = "</s>" # End Of Sentence token

def load_and_prepare_data(csv_path):
    """Loads data from CSV, preprocesses, and formats it for instruction finetuning."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {csv_path} was not found. Please ensure it's in the correct path.")
        

    if 'Star' not in df.columns or 'Comment' not in df.columns:
        raise ValueError("CSV must contain 'Star' and 'Comment' columns.")

    # Define sentiment label
    df['sentiment_label'] = df['Star'].apply(lambda x: "positive" if x >= 3 else "negative")

    # Format data
    formatted_texts = []
    for _, row in df.iterrows():
        if pd.isna(row['Comment']): # Handle potential NaN comments
            print(f"Warning: Skipping row with ID {row.get('ID', 'N/A')} due to missing comment.")
            continue
        text = ALPACA_PROMPT.format(str(row['Comment']), row['sentiment_label']) + EOS_TOKEN
        formatted_texts.append(text)

    dataset = Dataset.from_dict({"text": formatted_texts})

    # Split into train and test (optional, but good practice)
    # Using a small test size for demonstration
    if len(dataset) > 10: # Ensure enough data for a split
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })
        print(f"Data loaded: {len(dataset_dict['train'])} training examples, {len(dataset_dict['test'])} testing examples.")
        # print(f"Training data example:\n{dataset_dict['train'][0]['text'][:300]}...") # Print a snippet
    elif len(dataset) > 0:
        dataset_dict = DatasetDict({'train': dataset}) # Use all data for training if too small to split
        print(f"Data loaded: {len(dataset_dict['train'])} training examples (dataset too small to split for test).")
        # print(f"Training data example:\n{dataset_dict['train'][0]['text'][:300]}...")
    else:
        print("Error: No data loaded after processing. Check CSV content and formatting.")
        return None


    return dataset_dict

def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Error: CUDA (GPU) is not available. Unsloth requires a GPU.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    if num_gpus == 0:
        print("Error: No GPUs detected by PyTorch.")
        return
    # Unsloth typically works best when Accelerate handles device placement.
    # Manual device setting is usually not needed with `accelerate launch`.

    # 1. Load and prepare data
    print("Loading and preparing data...")
    dataset_dict = load_and_prepare_data(CSV_FILE_PATH)
    if dataset_dict is None or 'train' not in dataset_dict or len(dataset_dict['train']) == 0:
        print("Failed to load or process data. Exiting.")
        return

    # 2. Load model and tokenizer with Unsloth
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Autodetect, will be float16 on V100
        load_in_4bit=LOAD_IN_4BIT,
        # token="hf_YOUR_HUGGINGFACE_TOKEN", # Add if model is gated
    )

    # Add pad token if missing (common for some models like Llama)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Configure LoRA
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth", # Recommended by Unsloth
        random_state=42,
        # use_rslora=False, # Rank-Stabilized LoRA
        # loftq_config=None, # LoFTQ
    )
    print("LoRA configured.")
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Trainable GQA/LoRA parameters: {trainable_params:,d}")
    print(f"All parameters: {all_param:,d}")
    print(f"Percentage trainable: {(trainable_params / all_param * 100):.4f}%")


    # 4. Set up Training Arguments
    # These arguments are for a multi-GPU setup.
    # `accelerate launch` will handle the distribution.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        # max_steps=100, # For debugging: set a small number of steps
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),  # True for V100
        bf16=torch.cuda.is_bf16_supported(),      # False for V100
        logging_steps=5,                           # Log training loss frequently
        optim=OPTIMIZER,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="epoch",                     # Save a checkpoint every epoch
        report_to="mlflow", # Optional: if you use Weights & Biases
        ddp_find_unused_parameters=False, # May be needed for DDP, but Accelerate usually handles this.
                                          # Unsloth recommends FSDP if available/compatible.
    )

    # 5. Initialize Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict.get('test'), # Optional: if you have a test set
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=os.cpu_count() // 2 if os.cpu_count() else 1, # Number of CPU cores for dataset processing
        packing=False,  # Set to True if you want to pack short sequences for efficiency, but can be tricky.
        args=training_args,
    )
    print("Trainer initialized.")

    # 6. Start Training
    print("Starting training...")
    # Before training, it's good practice to show GPU memory summary if Unsloth provides it
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"Used memory (before training) = {start_gpu_memory} GB.")

    training_results = trainer.train()

    end_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_gpu_memory = round(end_gpu_memory - start_gpu_memory, 3)
    print(f"Used memory (during training) = {used_gpu_memory} GB.")
    print(f"Training finished. Results: {training_results.metrics}")


    # 7. Save the LoRA adapter
    print(f"Saving LoRA adapter to {LORA_ADAPTER_DIR}...")
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR) # Save tokenizer for easy loading
    print("LoRA adapter saved.")

    # To save a fully merged model (optional, requires more disk space and time):
    # print("Merging LoRA adapter and saving full model...")
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
    # print("Full model saved.")

    print("Script finished successfully!")

def run_inference(comment_text, adapter_path=LORA_ADAPTER_DIR):
    """Loads the finetuned LoRA model and runs inference on a given comment."""
    print("\n--- Running Inference ---")
    if not os.path.exists(adapter_path):
        print(f"Error: LoRA adapter not found at {adapter_path}. Please train the model first.")
        return

    # Load the base model and tokenizer again (as if in a new session)
    model_inf, tokenizer_inf = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, # Must be the same base model used for training
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if tokenizer_inf.pad_token is None:
        tokenizer_inf.pad_token = tokenizer_inf.eos_token

    # Load the LoRA adapter
    print(f"Loading LoRA adapter from {adapter_path} for inference...")
    FastLanguageModel.for_inference(model_inf) # Prepare model for inference
    model_inf.load_adapter(adapter_path)
    print("Adapter loaded.")

    # Prepare the input prompt
    # The response part is empty for the model to fill
    prompt = ALPACA_PROMPT.format(comment_text, "")

    inputs = tokenizer_inf(
        [prompt],
        return_tensors="pt",
        padding=True, # Pad to max_length or longest in batch
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Input comment: {comment_text}")
    print("Generating prediction...")

    # Generate the response
    outputs = model_inf.generate(
        **inputs,
        max_new_tokens=10,  # We only need "positive" or "negative"
        eos_token_id=tokenizer_inf.eos_token_id,
        pad_token_id=tokenizer_inf.pad_token_id,
        do_sample=False, # For classification, usually no sampling
        # temperature=0.1, # Can be used if do_sample=True
        # top_p=0.9,
    )

    # Decode only the newly generated tokens
    decoded_output = tokenizer_inf.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"Raw generated output: '{decoded_output}'")

    # Basic parsing of the output
    if "positive" in decoded_output.lower():
        predicted_sentiment = "positive"
    elif "negative" in decoded_output.lower():
        predicted_sentiment = "negative"
    else:
        predicted_sentiment = f"unknown (raw: {decoded_output})"

    print(f"Predicted Sentiment: {predicted_sentiment}")
    print("--- Inference Finished ---")
    return predicted_sentiment


if __name__ == "__main__":
    # --- Training Phase ---
    # This script is intended to be run with `accelerate launch your_script_name.py`
    # Ensure you have configured Accelerate first by running `accelerate config`.
    # For 8 GPUs, you would typically use `accelerate launch --num_processes=8 your_script_name.py`

    main()

    # --- Inference Phase (Example) ---
    # You can run this part after training, or in a separate script.
    # Make sure the LORA_ADAPTER_DIR points to your saved adapter.
    if os.path.exists(LORA_ADAPTER_DIR): # Check if training produced an adapter
        print("\nStarting inference example after training...")
        example_comment_positive = "这部电影太棒了！特效惊人，故事情节也非常吸引人。"
        example_comment_negative = "非常失望，演员表现很差，剧情也毫无逻辑可言。"

        run_inference(example_comment_positive)
        run_inference(example_comment_negative)

        # Test with one of the comments from the provided CSV example data
        # "连奥创都知道整容要去韩国。" (Star: 3, expected: positive)
        example_from_csv = "连奥创都知道整容要去韩国。"
        run_inference(example_from_csv)

        # "奥创弱爆了弱爆了弱爆了啊！！！！！！" (Star: 2, expected: negative)
        example_from_csv_2 = "奥创弱爆了弱爆了弱爆了啊！！！！！！"
        run_inference(example_from_csv_2)
    else:
        print(f"\nLoRA adapter not found at {LORA_ADAPTER_DIR}. Skipping inference example.")
        print("Please run the training first, or ensure LORA_ADAPTER_DIR is correctly set.")

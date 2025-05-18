from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from utils import predict_sentiment


model_name = "Qwen/Qwen3-30B-A3B"
model_id = "Qwen/Qwen3-30B-A3B"
num_labels = 2  # Or len(df['label'].unique())


base_model_id = model_id  # The original Qwen model ID
adapter_path = "results/final_lora_adapters"  # Path to your saved LoRA adapters

# Load the base model with 4-bit quantization (if you trained with it)
bnb_config_inference = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Or torch.float16
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    num_labels=num_labels,  # Ensure this matches
    quantization_config=bnb_config_inference,  # Apply quantization if adapters were trained on a quantized model
    trust_remote_code=True,
    device_map="auto",
)

# Load the LoRA adapters onto the base model
inference_model = PeftModel.from_pretrained(base_model, adapter_path)
# If you saved a merged model, you would load it directly:
# inference_model = AutoModelForSequenceClassification.from_pretrained(f"{output_dir}/final_merged_model", device_map="auto", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path
)  # Load tokenizer from the same place
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inference_model.eval()  # Set to evaluation mode

test_text_positive = "This is a fantastic product, I love it!"
test_text_negative = "What a terrible experience, I am very disappointed."

print(
    f"Sentiment for '{test_text_positive}': {predict_sentiment(test_text_positive, inference_model, tokenizer)}"
)
print(
    f"Sentiment for '{test_text_negative}': {predict_sentiment(test_text_negative, inference_model, tokenizer)}"
)

# Evaluate on a test set if you have one
# You can use the Trainer's evaluate method or write a custom loop
# test_dataset = ... (load and tokenize your test data similar to validation)
# results = trainer.evaluate(eval_dataset=test_dataset)
# print("Test set evaluation results:", results)
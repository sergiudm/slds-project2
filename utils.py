import pandas as pd
import torch
from datasets import Dataset, DatasetDict


# Apply formatting if needed (for SFTTrainer with 'text' and 'label' columns, this might not be necessary to change much)
# If you need a specific prompt format for generation-style fine-tuning:
# formatted_train_dataset = dataset_dict['train'].map(lambda x: {"text_formatted": f"Instruction: Classify sentiment.\nInput: {x['text']}\nOutput: {x['label']}"})
# formatted_validation_dataset = dataset_dict['validation'].map(lambda x: {"text_formatted": f"Instruction: Classify sentiment.\nInput: {x['text']}\nOutput: {x['label']}"})
# Then you would tell SFTTrainer to use "text_formatted" as the dataset_text_field.

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id # e.g., 0 or 1


def load_and_prepare_data(filepath="douban_movie.csv"):
    """
    Loads the dataset, handles missing values, and creates the target variable.
    A movie is considered 'loved' (1) if Star >= 4 (original was >=3, changed to 4 for better class separation potentially),
    otherwise 'not loved' (0).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Please check the path.")
        return None

    df = df[["Comment", "Star"]].copy()
    df.dropna(subset=["Comment", "Star"], inplace=True)
    df["Star"] = pd.to_numeric(df["Star"], errors="coerce")
    df.dropna(subset=["Star"], inplace=True)

    # Define 'Sentiment': 1 if Star >= 3 (Loved), 0 otherwise (Not Loved)
    # Adjusted threshold to 3 stars for 'loved' as it's a common practice.
    # Original prompt had "loved by the audience", 3, 4 & 5 stars usually represent this.
    df["Sentiment"] = df["Star"].apply(lambda x: 1 if x >= 3 else 0)

    print(f"Loaded {len(df)} reviews.")
    print(f"Class distribution:\n{df['Sentiment'].value_counts(normalize=True)}")

    # Return only the Comment and Sentiment columns
    return df[["Comment", "Sentiment"]]


if __name__ == "__main__":
    df = load_and_prepare_data()
    print('Data loaded and prepared:')
    print(df.head())
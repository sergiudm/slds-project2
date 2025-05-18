import csv
import json
import random
import math

def csv_to_json_converter(csv_file_path, json_file_path):
    """
    Converts a CSV file to a JSON file with a "prompt" (Comment) and 
    a "completion" (binary target variable: 1 if Star >= 3, else 0).
    Handles missing or invalid 'Comment' and 'Star' values by skipping rows.

    Args:
        csv_file_path (str): The path to the input CSV file.
        json_file_path (str): The path to the output JSON file.
    Returns:
        list: A list of processed dictionaries, or None if a major error occurs.
    """
    processed_data_list = []
    skipped_rows_count = 0

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            if 'Comment' not in csv_reader.fieldnames or 'Star' not in csv_reader.fieldnames:
                print(f"Error: CSV file must contain 'Comment' and 'Star' columns.")
                return None

            for row_number, row in enumerate(csv_reader, 1):
                comment_text = row.get('Comment', '').strip()
                star_text = row.get('Star', '').strip()

                # Validate Comment
                if not comment_text:
                    # print(f"Warning: Row {row_number}: Skipping due to missing or empty 'Comment'.")
                    skipped_rows_count += 1
                    continue

                # Validate and convert Star
                if not star_text:
                    # print(f"Warning: Row {row_number}: Skipping due to missing or empty 'Star'.")
                    skipped_rows_count += 1
                    continue
                
                try:
                    star_value = int(float(star_text)) # float() handles "3.0", int() converts to integer
                except ValueError:
                    # print(f"Warning: Row {row_number}: Skipping due to invalid 'Star' value: '{star_text}'. Not a number.")
                    skipped_rows_count += 1
                    continue

                # Create target variable
                target_variable = 1 if star_value >= 3 else 0
                
                json_object = {
                    "prompt": comment_text,
                    "completion": target_variable
                }
                processed_data_list.append(json_object)

        if skipped_rows_count > 0:
            print(f"Info: Skipped {skipped_rows_count} rows due to missing/invalid 'Comment' or 'Star' values during CSV conversion.")

        if not processed_data_list:
            print("Warning: No valid data was processed from the CSV file.")
            # Still save an empty list to the JSON to avoid downstream errors if expected
            # or return None if preferred. For now, let's save an empty list.
            
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(processed_data_list, json_file, ensure_ascii=False, indent=4)
        
        print(f"Successfully converted '{csv_file_path}' to '{json_file_path}' with {len(processed_data_list)} processed entries.")
        return processed_data_list

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during CSV to JSON conversion: {e}")
        return None

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the data into training, validation, and test sets.

    Args:
        data (list): A list of data items (dictionaries).
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        seed (int): Random seed for shuffling to ensure reproducibility.

    Returns:
        tuple: A tuple containing (train_data, val_data, test_data).
               Returns (None, None, None) if ratios don't sum to 1 or data is empty/None.
    """
    if not data: # Handles both empty list and None
        print("Error: No data provided to split.")
        return None, None, None

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print(f"Error: Ratios must sum to 1.0. Current sum: {train_ratio + val_ratio + test_ratio}")
        return None, None, None

    random.seed(seed)
    shuffled_data = data[:] 
    random.shuffle(shuffled_data)

    total_items = len(shuffled_data)
    
    # Ensure calculations for split points are robust
    train_count = math.floor(total_items * train_ratio)
    val_count = math.floor(total_items * val_ratio)
    # test_count is the remainder to ensure all data is used and counts sum up correctly
    
    # Handle potential floating point inaccuracies for the last split by assigning the remainder
    if total_items > 0:
        train_data = shuffled_data[:train_count]
        val_data = shuffled_data[train_count : train_count + val_count]
        test_data = shuffled_data[train_count + val_count:]
    else: # if total_items is 0
        train_data, val_data, test_data = [], [], []


    print(f"Data split: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test items.")
    # Verify that sum of splits equals total items
    if len(train_data) + len(val_data) + len(test_data) != total_items:
        print(f"Warning: Data split counts ({len(train_data) + len(val_data) + len(test_data)}) do not sum to total items ({total_items}). Check split logic.")

    return train_data, val_data, test_data

def save_json_data(data, file_path):
    """
    Saves a list of dictionaries to a JSON file.

    Args:
        data (list): The data to save. Can be None.
        file_path (str): The path to the output JSON file.
    """
    if data is None:
        print(f"No data to save for '{file_path}'. File will not be created/overwritten.")
        return
        
    try:
        with open(file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Successfully saved data to '{file_path}'")
    except Exception as e:
        print(f"An error occurred while saving '{file_path}': {e}")

# --- Main execution ---
# Define file paths
csv_file = 'assets/douban_movie.csv' # New name to reflect modified content for testing
intermediate_json_file = 'data/output_movies_processed.json'

train_json_file = 'data/train_data.json'
val_json_file = 'data/val_data.json'
test_json_file = 'data/test_data.json'

# 1. Convert CSV to the intermediate JSON file with processing
all_processed_data = csv_to_json_converter(csv_file, intermediate_json_file)

if all_processed_data: # Check if data processing was successful and returned data
    # 2. Split the processed data
    train_data, val_data, test_data = split_data(
        all_processed_data, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        seed=42 
    )

    # 3. Save the split datasets
    save_json_data(train_data, train_json_file)
    save_json_data(val_data, val_json_file)
    save_json_data(test_data, test_json_file)

    # Optional: Print counts and samples
    print(f"\nTotal processed entries: {len(all_processed_data)}")
    if train_data:
        print(f"Training samples: {len(train_data)}")
        print(f"Sample of '{train_json_file}':")
        print(json.dumps(train_data[:min(2, len(train_data))], ensure_ascii=False, indent=4))

    if val_data:
        print(f"Validation samples: {len(val_data)}")
        print(f"\nSample of '{val_json_file}':")
        print(json.dumps(val_data[:min(1, len(val_data))], ensure_ascii=False, indent=4))
        
    if test_data:
        print(f"Test samples: {len(test_data)}")
        print(f"\nSample of '{test_json_file}':")
        print(json.dumps(test_data[:min(1, len(test_data))], ensure_ascii=False, indent=4))

else:
    print("Skipping data splitting due to issues in CSV to JSON conversion or no valid data found.")
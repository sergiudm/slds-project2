import pandas as pd
import numpy as np
import re
import jieba  # For Chinese text segmentation
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import (
    MinMaxScaler,
)  # For scaling W2V/BERT for MultinomialNB if needed, though GaussianNB is preferred.
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import os

# --- Configuration ---
try:
    # The file should be in the same directory as the script, UTF-8 encoded, one stopword per line.
    with open("assets/stopwords_cn.txt", "r", encoding="utf-8") as f:
        loaded_stopwords = [line.strip() for line in f if line.strip()]
    if loaded_stopwords:
        CHINESE_STOPWORDS = set(loaded_stopwords)
        print(
            f"Successfully loaded {len(CHINESE_STOPWORDS)} stopwords from stopwords_cn.txt"
        )
    else:
        print("Warning: stopwords_cn.txt is empty. Using default stopwords list.")
except FileNotFoundError:
    raise FileNotFoundError(
        "Warning: stopwords_cn.txt not found. Using default stopwords list."
    )
except Exception as e:
    raise Exception(
        f"Warning: Error loading stopwords_cn.txt: {e}. Using default stopwords list."
    )


# --- 1. Load and Prepare Data ---
def load_and_prepare_data(filepath="assets/douban_movie.csv"):
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

    return df[["Comment", "Sentiment"]]


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    text = text.lower()
    words = jieba.lcut(text)
    words = [word for word in words if word not in CHINESE_STOPWORDS and len(word) > 1]
    return " ".join(words)


# --- 3. Text Vectorization ---
# TF-IDF
def vectorize_text_tf_idf(
    texts_train, texts_test, max_features=5000, min_df=5, max_df=0.7
):
    vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=min_df, max_df=max_df
    )
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)
    print(f"TF-IDF: Vectorized to {X_train_tfidf.shape[1]} features.")
    return X_train_tfidf, X_test_tfidf, vectorizer


# Word2Vec
def vectorize_text_word2vec(
    texts_train, texts_test, vector_size=100, window=5, min_count=5, workers=4, sg=1
):
    train_tokens = [text.split() for text in texts_train]
    test_tokens = [text.split() for text in texts_test]

    # Train Word2Vec model on training data only
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )  # sg=1 for skip-gram
    print("Word2Vec model training complete.")

    def document_vector(doc_tokens, model_wv, num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        n_words = 0
        # Index_to_key is a list of words in the vocabulary
        index2word_set = set(model_wv.index_to_key)
        for word in doc_tokens:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model_wv[word])
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    X_train_w2v = np.array(
        [document_vector(doc, w2v_model.wv, vector_size) for doc in train_tokens]
    )
    X_test_w2v = np.array(
        [document_vector(doc, w2v_model.wv, vector_size) for doc in test_tokens]
    )

    if X_train_w2v.shape[0] == 0 or X_train_w2v.shape[1] == 0:
        print(
            "Warning: Word2Vec resulted in empty training vectors. This might be due to a small vocabulary after preprocessing or small dataset."
        )
        return np.array([]), np.array([]), w2v_model  # Return empty arrays

    print(f"Word2Vec: Vectorized to {X_train_w2v.shape[1]} features.")
    return X_train_w2v, X_test_w2v, w2v_model


# BERT
def vectorize_text_bert(
    texts_list, model_name="bert-base-chinese", max_length=128, batch_size=16
):
    """
    Converts a list of preprocessed texts into BERT embeddings.
    Uses [CLS] token embedding.
    """
    print(f"Loading BERT model ({model_name}) and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    all_embeddings = []
    print(f"Generating BERT embeddings using {device}...")
    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded_input)

        # Use the [CLS] token embedding
        cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(texts_list):
            print(
                f"  Processed batch {i // batch_size + 1}/{(len(texts_list) - 1) // batch_size + 1}..."
            )

    final_embeddings = np.vstack(all_embeddings)
    print(f"BERT: Vectorized to {final_embeddings.shape[1]} features.")
    return (
        final_embeddings,
        model,
    )  # Return embeddings and the model (or just tokenizer if model not needed later)


# --- 4. Model Training & Evaluation ---
def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="Model",
    vectorization_method="Unknown",
):
    print(f"\n--- Training: {model_name} with {vectorization_method} ---")

    # Handle cases where X_train might be empty (e.g. Word2Vec failing)
    if X_train.shape[0] == 0 or (
        hasattr(X_train, "nnz") and X_train.nnz == 0
    ):  # check for sparse matrix emptiness too
        print(
            f"Skipping {model_name} with {vectorization_method} due to empty training data."
        )
        return None, {
            "vectorization": vectorization_method,
            "model": model_name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "cv_f1_macro": np.nan,
            "error": "Empty training data",
        }

    # Cross-validation
    try:
        # GaussianNB doesn't have predict_proba before fitting if X_train is empty,
        # and cross_val_score might try to use it.
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="f1_macro"
        )  # Reduced to 3 folds for speed
        cv_f1_macro = np.mean(cv_scores)
        print(
            f"Average 3-Fold CV F1-score (macro): {cv_f1_macro:.4f} (+/- {np.std(cv_scores):.4f})"
        )
    except Exception as e:
        print(
            f"Cross-validation failed for {model_name} with {vectorization_method}: {e}"
        )
        cv_f1_macro = np.nan

    # Training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    # Use pos_label=1 for binary classification metrics, focusing on the 'Loved' class
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("Test Set Performance (Positive Class: Loved):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Not Loved (0)", "Loved (1)"], zero_division=0
        )
    )

    metrics = {
        "vectorization": vectorization_method,
        "model": model_name,
        "accuracy": accuracy,
        "precision_loved": precision,  # Precision for 'Loved' class
        "recall_loved": recall,  # Recall for 'Loved' class
        "f1_loved": f1,  # F1 for 'Loved' class
        "cv_f1_macro": cv_f1_macro,
    }
    return model, metrics


# --- Main Execution ---
if __name__ == "__main__":
    # Create results directory
    os.makedirs("./results", exist_ok=True)

    all_results_summary = []

    # Load and prepare data
    data_df = load_and_prepare_data("douban_movie.csv")

    if data_df is not None and not data_df.empty:
        print("\nPreprocessing text data... (This may take a while for large datasets)")
        data_df_processed = data_df.copy()
        data_df_processed["Processed_Comment"] = data_df_processed["Comment"].apply(
            preprocess_text
        )
        data_df_processed = data_df_processed[
            data_df_processed["Processed_Comment"].str.strip().astype(bool)
        ]
        print(
            f"Number of reviews after preprocessing and removing empty comments: {len(data_df_processed)}"
        )

        if len(data_df_processed) < 50:  # Increased minimum threshold
            print(
                "Not enough data to proceed after preprocessing (less than 50 reviews). Exiting."
            )
        else:
            X_texts = data_df_processed[
                "Processed_Comment"
            ].tolist()  # Use list for BERT
            y_labels = data_df_processed["Sentiment"].values

            # Split data into training and testing sets (raw text for now)
            # We pass raw text lists to vectorizers which handle train/test split internally for fit/transform
            X_train_raw_texts, X_test_raw_texts, y_train, y_test = train_test_split(
                X_texts, y_labels, test_size=0.25, random_state=42, stratify=y_labels
            )
            print(
                f"\nTraining set size: {len(X_train_raw_texts)}, Test set size: {len(X_test_raw_texts)}"
            )

            # --- Vectorization Method 1: TF-IDF ---
            print("\n\n--- Starting TF-IDF Vectorization and Modeling ---")
            X_train_tfidf, X_test_tfidf, _ = vectorize_text_tf_idf(
                X_train_raw_texts, X_test_raw_texts
            )

            # Logistic Regression with TF-IDF
            lr_tfidf_model = LogisticRegression(
                solver="liblinear",
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            )
            _, metrics_lr_tfidf = train_and_evaluate_model(
                lr_tfidf_model,
                X_train_tfidf,
                y_train,
                X_test_tfidf,
                y_test,
                "Logistic Regression",
                "TF-IDF",
            )
            if metrics_lr_tfidf:
                all_results_summary.append(metrics_lr_tfidf)

            # Multinomial Naive Bayes with TF-IDF
            nb_multi_tfidf_model = MultinomialNB(alpha=0.1)  # Smoothing parameter
            _, metrics_nb_multi_tfidf = train_and_evaluate_model(
                nb_multi_tfidf_model,
                X_train_tfidf,
                y_train,
                X_test_tfidf,
                y_test,
                "Multinomial NB",
                "TF-IDF",
            )
            if metrics_nb_multi_tfidf:
                all_results_summary.append(metrics_nb_multi_tfidf)

            # --- Vectorization Method 2: Word2Vec ---
            print("\n\n--- Starting Word2Vec Vectorization and Modeling ---")
            # Note: Word2Vec min_count might need adjustment based on vocab size after preprocessing
            X_train_w2v, X_test_w2v, _ = vectorize_text_word2vec(
                X_train_raw_texts, X_test_raw_texts, min_count=2
            )

            if (
                X_train_w2v.ndim == 2
                and X_train_w2v.shape[0] > 0
                and X_train_w2v.shape[1] > 0
            ):
                # Logistic Regression with Word2Vec
                lr_w2v_model = LogisticRegression(
                    solver="liblinear",
                    random_state=42,
                    class_weight="balanced",
                    max_iter=1000,
                )
                _, metrics_lr_w2v = train_and_evaluate_model(
                    lr_w2v_model,
                    X_train_w2v,
                    y_train,
                    X_test_w2v,
                    y_test,
                    "Logistic Regression",
                    "Word2Vec",
                )
                if metrics_lr_w2v:
                    all_results_summary.append(metrics_lr_w2v)

                # Gaussian Naive Bayes with Word2Vec
                nb_gauss_w2v_model = GaussianNB()
                _, metrics_nb_gauss_w2v = train_and_evaluate_model(
                    nb_gauss_w2v_model,
                    X_train_w2v,
                    y_train,
                    X_test_w2v,
                    y_test,
                    "Gaussian NB",
                    "Word2Vec",
                )
                if metrics_nb_gauss_w2v:
                    all_results_summary.append(metrics_nb_gauss_w2v)
            else:
                print(
                    "Skipping Word2Vec based models due to issues in vectorization (e.g., empty vectors)."
                )
                for model_name, nb_model_type in [
                    ("Logistic Regression", None),
                    ("Gaussian NB", GaussianNB),
                ]:
                    all_results_summary.append(
                        {
                            "vectorization": "Word2Vec",
                            "model": model_name,
                            "accuracy": np.nan,
                            "precision_loved": np.nan,
                            "recall_loved": np.nan,
                            "f1_loved": np.nan,
                            "cv_f1_macro": np.nan,
                            "error": "Word2Vec vectorization failed or produced empty results.",
                        }
                    )

            # # --- Vectorization Method 3: BERT ---
            # print(
            #     "\n\n--- Starting BERT Vectorization and Modeling (This can be slow!) ---"
            # )
            # # BERT embeddings are generated for the combined dataset then split to ensure consistency if needed,
            # # but here we'll do it on pre-split raw texts for simplicity matching other vectorizers.
            # # However, for BERT, it's often better to tokenize and get embeddings once on full dataset if memory allows,
            # # then split. For this script, we'll stick to vectorizing train/test texts separately.

            # print("Generating BERT embeddings for training data...")
            # X_train_bert, _ = vectorize_text_bert(
            #     X_train_raw_texts
            # )  # BERT model itself not needed later for this script
            # print("Generating BERT embeddings for test data...")
            # X_test_bert, _ = vectorize_text_bert(X_test_raw_texts)

            # if (
            #     X_train_bert.ndim == 2
            #     and X_train_bert.shape[0] > 0
            #     and X_train_bert.shape[1] > 0
            # ):
            #     # Logistic Regression with BERT
            #     lr_bert_model = LogisticRegression(
            #         solver="liblinear",
            #         random_state=42,
            #         class_weight="balanced",
            #         max_iter=2000,
            #     )  # Increased max_iter
            #     _, metrics_lr_bert = train_and_evaluate_model(
            #         lr_bert_model,
            #         X_train_bert,
            #         y_train,
            #         X_test_bert,
            #         y_test,
            #         "Logistic Regression",
            #         "BERT",
            #     )
            #     if metrics_lr_bert:
            #         all_results_summary.append(metrics_lr_bert)

            #     # Gaussian Naive Bayes with BERT
            #     nb_gauss_bert_model = GaussianNB()
            #     _, metrics_nb_gauss_bert = train_and_evaluate_model(
            #         nb_gauss_bert_model,
            #         X_train_bert,
            #         y_train,
            #         X_test_bert,
            #         y_test,
            #         "Gaussian NB",
            #         "BERT",
            #     )
            #     if metrics_nb_gauss_bert:
            #         all_results_summary.append(metrics_nb_gauss_bert)
            # else:
            #     print("Skipping BERT based models due to issues in vectorization.")
            #     for model_name in ["Logistic Regression", "Gaussian NB"]:
            #         all_results_summary.append(
            #             {
            #                 "vectorization": "BERT",
            #                 "model": model_name,
            #                 "accuracy": np.nan,
            #                 "precision_loved": np.nan,
            #                 "recall_loved": np.nan,
            #                 "f1_loved": np.nan,
            #                 "cv_f1_macro": np.nan,
            #                 "error": "BERT vectorization failed or produced empty results.",
            #             }
            #         )

            # --- Save and Print Results Summary ---
            if all_results_summary:
                results_df = pd.DataFrame(all_results_summary)
                results_df = results_df.sort_values(
                    by=["vectorization", "model"]
                ).reset_index(drop=True)

                print("\n\n--- Overall Results Summary ---")
                print(results_df)

                output_path = os.path.join(
                    "./results/task2", "sentiment_analysis_comparison.csv"
                )
                try:
                    results_df.to_csv(
                        output_path, index=False, encoding="utf-8-sig"
                    )  # utf-8-sig for Excel compatibility with Chinese
                    print(f"\nResults successfully saved to: {output_path}")
                except Exception as e:
                    print(f"\nError saving results to CSV: {e}")
            else:
                print("\nNo results were generated to summarize or save.")
    else:
        print(
            "Could not proceed due to data loading issues or insufficient data after preprocessing."
        )

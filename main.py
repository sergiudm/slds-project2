import pandas as pd
import numpy as np
import re
import jieba # For Chinese text segmentation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import subprocess

# --- Configuration ---
try:
    with open("./stopwords_cn.txt", "r", encoding="utf-8") as f:
        CHINESE_STOPWORDS = [line.strip() for line in f if line.strip()]
    if not CHINESE_STOPWORDS:
        raise ValueError("Stopwords list is empty.")
except Exception as e:
    raise Exception(f"Warning: Error loading stopwords_cn.txt: {e}. Using a default small list.")


# --- 1. Load and Prepare Data ---
def load_and_prepare_data(filepath="douban_movie.csv"):
    """
    Loads the dataset, handles missing values, and creates the target variable.
    A movie is considered 'loved' (1) if Star >= 3, otherwise 'not loved' (0).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Please check the path.")
        return None

    # Select relevant columns
    df = df[['Comment', 'Star']].copy()

    # Drop rows with missing comments or stars, as they are crucial
    df.dropna(subset=['Comment', 'Star'], inplace=True)

    # Ensure 'Star' is numeric (it might be read as object if there are non-numeric entries)
    df['Star'] = pd.to_numeric(df['Star'], errors='coerce')
    df.dropna(subset=['Star'], inplace=True) # Drop rows where Star could not be converted

    # Define the target variable 'Sentiment'
    # Loved (1) if Star >= 3, Not Loved (0) if Star < 3
    df['Sentiment'] = df['Star'].apply(lambda x: 1 if x >= 3 else 0)

    print(f"Loaded {len(df)} reviews.")
    print(f"Class distribution:\n{df['Sentiment'].value_counts(normalize=True)}")
    
    return df[['Comment', 'Sentiment']]

# --- 2. Text Preprocessing ---
def preprocess_text(text):
    """
    Cleans and tokenizes Chinese text:
    - Removes special symbols and numbers (keeps Chinese characters and basic punctuation for context)
    - Converts to lowercase
    - Segments text using jieba
    - Removes stopwords
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    # Remove special characters, numbers, and English letters. Keep Chinese characters.
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text) # Keep only Chinese characters
    text = text.lower() # Convert to lowercase (though less critical for Chinese)

    # Tokenize using jieba
    words = jieba.lcut(text)

    # Remove stopwords and very short words (often noise)
    words = [word for word in words if word not in CHINESE_STOPWORDS and len(word) > 1]

    return " ".join(words) # Return space-separated tokens

# --- 3. Text Vectorization ---
def vectorize_text_TF_IDF(texts_train, texts_test, max_features=5000, min_df=5, max_df=0.7):
    """
    Converts preprocessed text into TF-IDF vectors.
    min_df: ignore terms that have a document frequency strictly lower than the given threshold.
    max_df: ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)
    
    print(f"TF-IDF: Vectorized to {X_train_tfidf.shape[1]} features.")
    return X_train_tfidf, X_test_tfidf, vectorizer

def vectorize_text_Word2Vec(texts_train, texts_test, vector_size=100, window=5, min_count=5):
    """
    Converts preprocessed text into Word2Vec vectors by averaging word vectors in each document.
    - vector_size: Dimensionality of the word vectors
    - window: Maximum distance between a target word and words around it
    - min_count: Ignores all words with total frequency lower than this
    """
    # Convert space-separated tokens back to lists for Word2Vec
    train_tokens = [text.split() for text in texts_train]
    test_tokens = [text.split() for text in texts_test]
    
    # Train Word2Vec model on training data only
    w2v_model = Word2Vec(sentences=train_tokens, vector_size=vector_size, window=window, 
                            min_count=min_count, workers=4, sg=1)  # sg=1 for skip-gram
    
    # Create document vectors by averaging word vectors for each document
    def document_vector(doc, model):
        # Filter out words not in vocabulary
        doc_words = [word for word in doc if word in model.wv.index_to_key]
        if not doc_words:
            # Return zeros if no words are in vocabulary
            return np.zeros(model.vector_size)
        # Return the average of word vectors
        return np.mean([model.wv[word] for word in doc_words], axis=0)
    
    # Convert each document to vector
    X_train_w2v = np.array([document_vector(doc, w2v_model) for doc in train_tokens])
    X_test_w2v = np.array([document_vector(doc, w2v_model) for doc in test_tokens])
    
    print(f"Word2Vec: Vectorized to {X_train_w2v.shape[1]} features.")
    return X_train_w2v, X_test_w2v, w2v_model


def vectorize_text_BERT(texts_train, texts_test, model_name="bert-base-chinese", max_length=128):
    """
    Converts preprocessed text into BERT embeddings.
    - texts_train/texts_test: Preprocessed text inputs
    - model_name: Pre-trained BERT model to use (default is Chinese BERT)
    - max_length: Maximum sequence length for BERT input
    
    Returns document embeddings from the [CLS] token of BERT's last hidden layer.
    """

    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Function to get BERT embeddings for a batch of texts
    def get_bert_embeddings(texts, batch_size=8):
        all_embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare input
            encoded_input = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Get model output (without gradient calculation)
            with torch.no_grad():
                output = model(**encoded_input)
            
            # Use the [CLS] token embedding as document embedding
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
            # Print progress occasionally
            if (i+batch_size) % 100 == 0 or i+batch_size >= len(texts):
                print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts...")
        
        # Concatenate all batch embeddings
        return np.vstack(all_embeddings)
    
    print("Generating BERT embeddings for training data...")
    X_train_bert = get_bert_embeddings(texts_train)
    
    print("Generating BERT embeddings for test data...")
    X_test_bert = get_bert_embeddings(texts_test)
    
    print(f"BERT: Vectorized to {X_train_bert.shape[1]} features.")
    return X_train_bert, X_test_bert, model

# --- 4. Model Training & Evaluation ---
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a model, performs 5-fold cross-validation, and evaluates on the test set.
    """
    print(f"\n--- {model_name} ---")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro') # f1_macro for potentially imbalanced classes
    print(f"Average 5-Fold CV F1-score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability for the positive class

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Test Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (Positive Class)")
    print(f"  Recall:    {recall:.4f} (Positive Class)")
    print(f"  F1-score:  {f1:.4f} (Positive Class)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Loved (0)', 'Loved (1)'], zero_division=0))
    
    return model, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "cv_f1": np.mean(cv_scores)}

# --- Main Execution ---
if __name__ == "__main__":
    # Load and prepare data
    data_df = load_and_prepare_data("douban_movie.csv")

    if data_df is not None and not data_df.empty:
        # Apply text preprocessing
        # Using .copy() to avoid SettingWithCopyWarning if 'Comment' is a slice
        print("\nPreprocessing text data... (This may take a while for large datasets)")
        data_df_processed = data_df.copy()
        data_df_processed['Processed_Comment'] = data_df_processed['Comment'].apply(preprocess_text)
        
        # Remove rows where 'Processed_Comment' is empty after preprocessing
        data_df_processed = data_df_processed[data_df_processed['Processed_Comment'].str.strip().astype(bool)]
        print(f"Number of reviews after preprocessing and removing empty comments: {len(data_df_processed)}")

        if len(data_df_processed) < 10: # Arbitrary small number, adjust as needed
             print("Not enough data to proceed after preprocessing. Exiting.")
        else:
            # Split data into training and testing sets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                data_df_processed['Processed_Comment'],
                data_df_processed['Sentiment'],
                test_size=0.25, # 25% for testing
                random_state=42, # For reproducibility
                stratify=data_df_processed['Sentiment'] # Important for imbalanced datasets
            )
            print(f"\nTraining set size: {len(X_train_raw)}, Test set size: {len(X_test_raw)}")

            # Vectorize text
            X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text_TF_IDF(X_train_raw, X_test_raw)
            X_train_word2vec, X_test_word2vec, w2v_model = vectorize_text_Word2Vec(X_train_raw, X_test_raw)
            X_train_bert, X_test_bert, bert_model = vectorize_text_BERT(X_train_raw, X_test_raw)

            # Initialize models
            log_reg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
            naive_bayes_model = MultinomialNB(alpha=0.1) # alpha for Laplace smoothing

            # Train and evaluate Logistic Regression
            trained_log_reg, log_reg_metrics = train_and_evaluate_model(
                log_reg_model, X_train_tfidf, y_train, X_test_tfidf, y_test, "Logistic Regression"
            )

            # Train and evaluate Naive Bayes
            trained_nb, nb_metrics = train_and_evaluate_model(
                naive_bayes_model, X_train_tfidf, y_train, X_test_tfidf, y_test, "Naive Bayes (Multinomial)"
            )

            # --- 5. Reasoning Analysis (Example) ---
            print("\n--- Reasoning Analysis ---")
            print("Model performance depends heavily on data quality, preprocessing, and feature engineering.")
            
            if log_reg_metrics and nb_metrics:
                print("\nComparison:")
                if log_reg_metrics['f1'] > nb_metrics['f1']:
                    print("Logistic Regression performed slightly better on the F1-score for the 'Loved' class.")
                elif nb_metrics['f1'] > log_reg_metrics['f1']:
                    print("Naive Bayes performed slightly better on the F1-score for the 'Loved' class.")
                else:
                    print("Both models had similar F1-scores for the 'Loved' class.")

                print("\nKey Observations:")
                print("- TF-IDF with a limited number of features (max_features=5000) was used. Increasing this might capture more nuances but also more noise.")
                print("- Stopword removal and keeping only Chinese characters are crucial preprocessing steps for this dataset.")
                print("- The class distribution (loved vs. not loved) can impact performance. `class_weight='balanced'` was used for Logistic Regression to mitigate this. Naive Bayes is generally less sensitive to imbalanced data but can still be affected.")
                print("- Cross-validation scores give an idea of how well the model generalizes. If test scores are much lower than CV scores, it might indicate overfitting to the training data subset used for final training.")
                print("- Precision and Recall for the 'Loved' class: High precision means that when the model predicts a movie is 'loved', it's likely correct. High recall means the model finds most of the 'loved' movies.")
                print("  - If Recall is low for 'Loved', it means the model misses many movies that are actually loved.")
                print("  - If Precision is low for 'Loved', it means many movies predicted as 'loved' are actually not.")
            
            print("\nFurther Potential Improvements:")
            print("- More sophisticated preprocessing: e.g., handling negations, using n-grams (tfidf_vectorizer can do this with `ngram_range`).")
            print("- Advanced vectorization: Word2Vec or BERT embeddings could capture semantic meaning better but are more complex to implement and train.")
            print("- Hyperparameter tuning for TF-IDF and the models (e.g., using GridSearchCV).")
            print("- Using a more comprehensive Chinese stopword list.")
            print("- Error analysis: Examining misclassified reviews to understand model weaknesses.")

            # Example of how to predict a new review (using the better model, e.g., Logistic Regression)
            # new_review_text = "这部电影真是太精彩了，剧情紧凑，演员表现也很棒！"
            # processed_new_review = preprocess_text(new_review_text)
            # vectorized_new_review = tfidf_vectorizer.transform([processed_new_review])
            # prediction = trained_log_reg.predict(vectorized_new_review)
            # prediction_proba = trained_log_reg.predict_proba(vectorized_new_review)
            # print(f"\nPrediction for new review '{new_review_text}':")
            # print(f"  Sentiment: {'Loved' if prediction[0] == 1 else 'Not Loved'}")
            # print(f"  Probability of being Loved: {prediction_proba[0][1]:.4f}")

    else:
        print("Could not proceed due to data loading issues.")


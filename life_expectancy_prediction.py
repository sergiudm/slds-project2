import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
# The target variable column name
TARGET_COLUMN = "Life expectancy at birth, total (years)"
# List of feature column names (12 features)
FEATURE_COLUMNS = [
    "Agriculture, forestry, and fishing, value added (% of GDP)",
    "Annual freshwater withdrawals, total (% of internal resources)",
    "Current health expenditure (% of GDP)",
    "Forest area (% of land area)",
    "GDP (current US$)",
    "Immunization, measles (% of children ages 12-23 months)",
    "Income share held by lowest 20%",
    "Industry (including construction), value added (% of GDP)",
    "Population, total",
    "Prevalence of underweight, weight for age (% of children under 5)",
    "Research and development expenditure (% of GDP)",
    "School enrollment, secondary (% net)",
]
COUNTRY_COLUMN = "Country Name"

FILE_PATH = "life_indicator_2008-2018.xlsx"
YEAR_TRAIN = 2008  # This will be used as the sheet name for training data
YEAR_TEST = 2018  # This will be used as the sheet name for testing data


def load_data(file_path, train_sheet_name, test_sheet_name):
    """Loads data from specified sheets in the Excel file."""
    df_train = None
    df_test = None

    print(f"Attempting to load training data from sheet: '{train_sheet_name}'")
    try:
        df_train = pd.read_excel(file_path, sheet_name=train_sheet_name)
        print(
            f"Training data from sheet '{train_sheet_name}' loaded successfully. Shape: {df_train.shape}"
        )
        print(f"\nFirst 5 rows of training data ({train_sheet_name}):")
        print(df_train.head())
        print(f"\nColumn names in training data ({train_sheet_name}):")
        print(df_train.columns.tolist())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return None, None
    except ValueError as e:  # Handles case where sheet_name is not found
        if f"Worksheet named '{train_sheet_name}' not found" in str(e):
            print(
                f"Error: Worksheet named '{train_sheet_name}' not found in '{file_path}'."
            )
        else:
            print(f"Error loading training sheet '{train_sheet_name}': {e}")
        return None, None
    except Exception as e:
        print(f"Error loading training sheet '{train_sheet_name}': {e}")
        return None, None

    print(f"\nAttempting to load testing data from sheet: '{test_sheet_name}'")
    try:
        df_test = pd.read_excel(file_path, sheet_name=test_sheet_name)
        print(
            f"Testing data from sheet '{test_sheet_name}' loaded successfully. Shape: {df_test.shape}"
        )
        print(f"\nFirst 5 rows of testing data ({test_sheet_name}):")
        print(df_test.head())
        print(f"\nColumn names in testing data ({test_sheet_name}):")
        print(df_test.columns.tolist())
    except (
        FileNotFoundError
    ):  # Should have been caught by train load, but good practice
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return df_train, None  # df_train might be loaded
    except ValueError as e:  # Handles case where sheet_name is not found
        if f"Worksheet named '{test_sheet_name}' not found" in str(e):
            print(
                f"Error: Worksheet named '{test_sheet_name}' not found in '{file_path}'."
            )
        else:
            print(f"Error loading testing sheet '{test_sheet_name}': {e}")
        return df_train, None
    except Exception as e:
        print(f"Error loading testing sheet '{test_sheet_name}': {e}")
        return df_train, None

    return df_train, df_test


def data_understanding(df, target_column, feature_columns, data_description="Dataset"):
    """Performs initial data understanding steps on a given DataFrame."""
    if df is None:
        print(
            f"Skipping data understanding for {data_description} as DataFrame is None."
        )
        return False

    print(f"\n--- a. Data Understanding for {data_description} ---")

    # Check if essential columns exist
    required_columns = feature_columns + [
        target_column
    ]  # YEAR_COLUMN is not in individual sheets
    missing_essential_cols = [col for col in required_columns if col not in df.columns]
    if missing_essential_cols:
        print(
            f"\nERROR: The following essential columns are missing from the {data_description}: {missing_essential_cols}"
        )
        print(
            "Please update TARGET_COLUMN and FEATURE_COLUMNS at the beginning of the script."
        )
        print(f"Available columns in {data_description} are: {df.columns.tolist()}")
        return False  # Indicate failure

    # 1. Visualize relationships between features and target using a heatmap
    print("\n1. Correlation Heatmap:")
    plt.figure(figsize=(12, 10))
    # Include the target variable in the correlation calculation
    # Ensure all columns for correlation are numeric; handle potential errors
    try:
        correlation_matrix = df[feature_columns + [target_column]].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title(f"Correlation Heatmap - {data_description}")
        plt.show()
    except Exception as e:
        print(f"Could not generate correlation heatmap for {data_description}: {e}")
        print("Ensure all feature columns and the target column are numeric.")

    # 2. Look at the distribution of life expectancy at birth
    print(f"\n2. Distribution of '{target_column}':")
    if target_column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_column].dropna(), kde=True)
        plt.title(f"Distribution of {target_column} - {data_description}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.show()
        print(df[target_column].describe())
    else:
        print(f"Target column '{target_column}' not found in {data_description}.")

    # 3. Deal with any missing data
    print("\n3. Missing Data Analysis:")
    # Check if all feature_columns and target_column exist before attempting to access them
    columns_to_check_missing = [
        col for col in feature_columns + [target_column] if col in df.columns
    ]
    if not columns_to_check_missing:
        print(
            f"No valid feature or target columns found in {data_description} for missing data analysis."
        )
        return True  # Not a failure of this step, but data might be unusable later

    missing_values = df[columns_to_check_missing].isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage (%)": missing_percentage}
    )
    missing_info = missing_info[missing_info["Missing Values"] > 0]

    if not missing_info.empty:
        print(missing_info)
        print(
            "\nNote: Median imputation will be used for missing numerical data during preprocessing."
        )
    else:
        print(
            f"No missing values found in specified features or target column for {data_description}."
        )
    return True  # Indicate success


def preprocess_data(X_train, X_test, y_train, y_test):
    """Handles missing data in features and scales features. Handles missing y_train."""

    # Impute missing values in features using Median
    imputer_X = SimpleImputer(strategy="median")

    X_train_imputed = imputer_X.fit_transform(X_train)
    X_test_imputed = imputer_X.transform(X_test)  # Use transform only, fitted on train

    X_train_imputed_df = pd.DataFrame(
        X_train_imputed, columns=X_train.columns, index=X_train.index
    )
    X_test_imputed_df = pd.DataFrame(
        X_test_imputed, columns=X_test.columns, index=X_test.index
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed_df)
    X_test_scaled = scaler.transform(X_test_imputed_df)  # Use transform only

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    # Impute missing values in y_train (target variable for training)
    # It's crucial that y_train does not have NaNs for model training.
    y_imputer = SimpleImputer(strategy="median")
    y_train_imputed = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_train_processed = pd.Series(
        y_train_imputed, index=y_train.index, name=y_train.name
    )

    # For y_test, rows with NaN target values cannot be used for evaluation.
    # We will filter them out along with corresponding X_test rows before evaluation.
    y_test_nan_mask = ~pd.Series(y_test).isna()  # Ensure y_test is a Series for .isna()
    y_test_cleaned = y_test[y_test_nan_mask]
    X_test_scaled_df_cleaned = X_test_scaled_df[y_test_nan_mask]

    # Ensure y_test_cleaned is a pandas Series with an index
    if not isinstance(y_test_cleaned, pd.Series):
        if hasattr(y_test, "index"):
            y_test_cleaned = pd.Series(
                y_test_cleaned, index=y_test[y_test_nan_mask].index
            )
        else:  # Fallback if original y_test had no index (e.g. numpy array)
            y_test_cleaned = pd.Series(y_test_cleaned)

    return (
        X_train_scaled_df,
        X_test_scaled_df_cleaned,
        y_train_processed,
        y_test_cleaned,
    )


def modeling_and_evaluation(X_train, y_train, X_test, y_test, feature_columns):
    """Trains different models, evaluates them, and identifies important features."""
    if X_train.empty or y_train.empty:
        print("Skipping modeling: Training data (X_train or y_train) is empty.")
        return None, {}
    if X_test.empty or y_test.empty:
        print(
            "Skipping modeling: Test data (X_test or y_test) is empty. This might be due to all target values in the test set being NaN."
        )
        return None, {}

    print("\n--- b. Modeling ---")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "Support Vector Regressor (SVR)": SVR(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[name] = {"MSE": mse, "R²": r2}
        trained_models[name] = model

        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Feature Importance (for tree-based models and linear models)
    print("\nFeature Importances:")

    lr_model = trained_models.get("Linear Regression")
    if lr_model and hasattr(lr_model, "coef_"):
        try:
            lr_coeffs = pd.Series(lr_model.coef_, index=feature_columns).sort_values(
                ascending=False
            )
            plt.figure(figsize=(10, 6))
            lr_coeffs.plot(kind="bar")
            plt.title("Feature Importances (Coefficients) - Linear Regression")
            plt.ylabel("Coefficient Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot Linear Regression coefficients: {e}")

    rf_model = trained_models.get("Random Forest Regressor")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        try:
            rf_importances = pd.Series(
                rf_model.feature_importances_, index=feature_columns
            ).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            rf_importances.plot(kind="bar")
            plt.title("Feature Importances - Random Forest Regressor")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.show()
            print("\nRandom Forest Feature Importances Determination:")
            print(
                "Calculated as the mean decrease in impurity (Gini importance) from splits on each feature."
            )
        except Exception as e:
            print(f"Could not plot Random Forest importances: {e}")

    gb_model = trained_models.get("Gradient Boosting Regressor")
    if gb_model and hasattr(gb_model, "feature_importances_"):
        try:
            gb_importances = pd.Series(
                gb_model.feature_importances_, index=feature_columns
            ).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            gb_importances.plot(kind="bar")
            plt.title("Feature Importances - Gradient Boosting Regressor")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.show()
            print("\nGradient Boosting Feature Importances Determination:")
            print(
                "Calculated based on how much each feature contributes to reducing the loss function."
            )
        except Exception as e:
            print(f"Could not plot Gradient Boosting importances: {e}")

    return trained_models, results


def analysis_of_predictions(
    model, X_test, y_test, model_name, target_column_name, test_year, output_dir="results/task1"
):
    """Visualizes prediction differences and errors for a given model."""
    if model is None:
        print(f"Skipping prediction analysis for {model_name}: Model is None.")
        return
    if X_test.empty or y_test.empty:
        print(f"Skipping prediction analysis for {model_name}: Test data is empty.")
        return

    print(f"\n--- c. Analysis of Predictions for {model_name} (Year {test_year}) ---")
    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel(f"Actual {target_column_name}")
    plt.ylabel(f"Predicted {target_column_name}")
    plt.title(f"Actual vs. Predicted Life Expectancy ({model_name} - {test_year})")
    plt.grid(True)
    plt.savefig(f"{output_dir}/actual_vs_predicted_{model_name}_{test_year}.png")

    errors = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Prediction Errors ({model_name} - {test_year})")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.savefig(f"{output_dir}/prediction_errors_{model_name}_{test_year}.png")

    print(f"\nSummary statistics for prediction errors ({model_name}):")
    print(errors.describe())

    mean_error = errors.mean()
    std_error = errors.std()
    outlier_threshold_upper = mean_error + 2 * std_error
    outlier_threshold_lower = mean_error - 2 * std_error

    # Ensure errors is a Pandas Series for boolean indexing
    if not isinstance(errors, pd.Series):
        errors = pd.Series(errors, index=y_test.index)

    outliers = errors[
        (errors > outlier_threshold_upper) | (errors < outlier_threshold_lower)
    ]
    if not outliers.empty:
        print(
            f"\nPotential Outliers (errors > 2 std dev from mean error) for {model_name}:"
        )
        outlier_indices = outliers.index

        # Reconstruct predictions for outlier_indices if predictions is a numpy array
        # This assumes X_test fed to predict() has the same index as y_test.
        # If X_test was filtered and re-indexed, this needs careful alignment.
        # Given current preprocess_data, X_test_scaled_df_cleaned and y_test_cleaned should align.

        # Find original indices in X_test that correspond to outlier_indices in y_test
        # This step is tricky if X_test was modified. Assuming X_test has the same index as y_test.

        # Create a temporary Series for predictions to align with y_test's index
        predictions_series = pd.Series(predictions, index=y_test.index)

        outlier_data = pd.DataFrame(
            {
                "Actual": y_test.loc[outlier_indices],
                "Predicted": predictions_series.loc[outlier_indices],
                "Error": outliers,
            }
        )
        # If COUNTRY_COLUMN was preserved and is in X_test.loc[outlier_indices], it could be added here.
        print(outlier_data)
        print("\nPossible reasons for outliers:")
        print(
            f"- Unique national circumstances in {test_year} not captured by {YEAR_TRAIN}-trained model."
        )
        print("- Data quality issues for specific countries/features.")
        print("- Model limitations for these specific cases.")
    else:
        print(
            "\nNo significant outliers (errors > 2 std dev from mean error) detected."
        )


def main():
    """Main function to run the life expectancy prediction task."""
    train_sheet_name = str(YEAR_TRAIN)
    test_sheet_name = str(YEAR_TEST)

    df_train_full, df_test_full = load_data(
        FILE_PATH, train_sheet_name, test_sheet_name
    )

    if df_train_full is None:
        print(
            f"\nExiting: Failed to load training data from sheet '{train_sheet_name}'."
        )
        print(
            "Please ensure the Excel file and sheet name are correct, and column names are configured."
        )
        return
    if df_test_full is None:
        print(
            f"\nWarning: Failed to load testing data from sheet '{test_sheet_name}'. Analysis will be limited."
        )
        # Depending on requirements, you might want to exit or proceed with training only.
        # For this script, we'll try to proceed if training data is okay, but testing steps will fail.

    # --- Data Understanding (on training data) ---
    print(f"\nPerforming data understanding for training data (Year {YEAR_TRAIN})...")
    if not data_understanding(
        df_train_full, TARGET_COLUMN, FEATURE_COLUMNS, f"Training Data ({YEAR_TRAIN})"
    ):
        print(
            "\nExiting due to issues in data understanding for training data (e.g., missing columns)."
        )
        print("Please check TARGET_COLUMN and FEATURE_COLUMNS configurations.")
        return

    # --- Prepare data for modeling ---
    # Check if target and features are in training data
    if TARGET_COLUMN not in df_train_full.columns or not all(
        col in df_train_full.columns for col in FEATURE_COLUMNS
    ):
        print(
            f"Error: Target column '{TARGET_COLUMN}' or some feature columns not found in training data sheet '{train_sheet_name}'."
        )
        print(f"Available columns: {df_train_full.columns.tolist()}")
        return

    # Drop rows where target is NaN for training, as they can't be used for supervised learning.
    df_train_full.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df_train_full.empty:
        print(
            f"Error: Training data is empty after dropping rows with NaN in '{TARGET_COLUMN}'."
        )
        return

    X_train = df_train_full[FEATURE_COLUMNS]
    y_train = df_train_full[TARGET_COLUMN]

    # Prepare test data if available
    X_test = pd.DataFrame()  # Initialize empty
    y_test = pd.Series(dtype="float64")  # Initialize empty

    if df_test_full is not None:
        if TARGET_COLUMN not in df_test_full.columns or not all(
            col in df_test_full.columns for col in FEATURE_COLUMNS
        ):
            print(
                f"Warning: Target column '{TARGET_COLUMN}' or some feature columns not found in testing data sheet '{test_sheet_name}'."
            )
            print(f"Available columns: {df_test_full.columns.tolist()}")
            print("Testing and prediction analysis will be skipped or limited.")
            df_test_full = None  # Mark as unusable for consistent X/y separation
        else:
            X_test = df_test_full[FEATURE_COLUMNS]
            y_test = df_test_full[
                TARGET_COLUMN
            ]  # y_test NaNs will be handled in preprocess_data
    else:  # df_test_full was None from load_data
        print(
            "Test data is not available. Modeling will proceed on training data only if possible, but no evaluation on test set."
        )

    if X_train.empty:
        print(
            f"ERROR: Training features (X_train) are empty after processing sheet '{train_sheet_name}'."
        )
        return

    # --- Preprocessing ---
    # Note: If df_test_full was None or invalid, X_test and y_test will be empty.
    # preprocess_data needs to handle this gracefully or we check before calling.
    if (
        X_test.empty and df_test_full is not None
    ):  # This implies columns were missing in test sheet
        print("Skipping preprocessing of test data as it's invalid or incomplete.")
        X_train_processed, _, y_train_processed, _ = preprocess_data(
            X_train, X_test.copy(), y_train, y_test.copy()
        )  # Pass copies of empty
        X_test_processed, y_test_processed = (
            pd.DataFrame(),
            pd.Series(dtype="float64"),
        )  # Ensure they are empty for downstream checks
    elif df_test_full is None:  # Test data was not loaded at all
        print("Test data not loaded. Preprocessing only training data.")
        # Impute and scale X_train
        imputer_X_train = SimpleImputer(strategy="median")
        X_train_imputed = imputer_X_train.fit_transform(X_train)
        X_train_imputed_df = pd.DataFrame(
            X_train_imputed, columns=X_train.columns, index=X_train.index
        )
        scaler_train = StandardScaler()
        X_train_scaled_arr = scaler_train.fit_transform(X_train_imputed_df)
        X_train_processed = pd.DataFrame(
            X_train_scaled_arr, columns=X_train.columns, index=X_train.index
        )

        # Impute y_train
        y_imputer_train = SimpleImputer(strategy="median")
        y_train_imputed_arr = y_imputer_train.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()
        y_train_processed = pd.Series(
            y_train_imputed_arr, index=y_train.index, name=y_train.name
        )

        X_test_processed, y_test_processed = (
            pd.DataFrame(),
            pd.Series(dtype="float64"),
        )  # Empty for downstream
    else:  # Both train and test data seem okay to proceed with standard preprocessing
        print("\nPreprocessing data (Imputation and Scaling)...")
        X_train_processed, X_test_processed, y_train_processed, y_test_processed = (
            preprocess_data(X_train, X_test, y_train, y_test)
        )

    if X_train_processed.empty or y_train_processed.empty:
        print(
            "Training data is empty after preprocessing. Cannot proceed with modeling."
        )
        return

    # --- Modeling and Evaluation ---
    # If X_test_processed or y_test_processed is empty, modeling_and_evaluation should handle it
    trained_models, results = modeling_and_evaluation(
        X_train_processed,
        y_train_processed,
        X_test_processed,
        y_test_processed,
        FEATURE_COLUMNS,
    )

    if trained_models and not X_test_processed.empty and not y_test_processed.empty:
        best_model_name = None
        best_r2 = -float("inf")
        if results:  # results might be empty if evaluation couldn't run
            for name, metrics in results.items():
                if metrics["R²"] > best_r2:
                    best_r2 = metrics["R²"]
                    best_model_name = name

        model_to_analyze_name = None
        if best_model_name and best_model_name in trained_models:
            model_to_analyze_name = best_model_name
            print(
                f"\nAnalyzing predictions for the best model based on R²: {model_to_analyze_name}"
            )
        elif "Random Forest Regressor" in trained_models:
            model_to_analyze_name = "Random Forest Regressor"
            print(
                f"\nNo best model by R² or results empty. Analyzing predictions for Random Forest Regressor by default."
            )

        if model_to_analyze_name:
            analysis_of_predictions(
                trained_models[model_to_analyze_name],
                X_test_processed,
                y_test_processed,
                model_to_analyze_name,
                TARGET_COLUMN,
                YEAR_TEST,
            )
        else:
            print(
                "\nNo suitable model available for detailed prediction analysis on test data."
            )
    elif trained_models:
        print(
            "\nModels were trained, but test data was not available or suitable for evaluation/prediction analysis."
        )
    else:
        print(
            "\nModeling step did not produce any models. Skipping prediction analysis."
        )

    print("\n--- End of Analysis ---")


if __name__ == "__main__":
    main()
    # Keep plots open until manually closed if script finishes quickly
    if plt.get_fignums():  # Check if any figures were created
        plt.show(block=True)

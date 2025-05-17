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

try:
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print(
        "Warning: mlxtend library not found. Stepwise Forward Selection will be skipped."
    )
    print("To enable this feature, please install mlxtend: pip install mlxtend")


# The target variable column name
TARGET_COLUMN = "Life expectancy at birth, total (years)"
# List of feature column names (initial 12 features)
# This list will be dynamically updated by engineer_features()
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
YEAR_TRAIN = 2008  #  for training data
YEAR_TEST = 2018  #  for testing data


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
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return df_train, None
    except ValueError as e:
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


def engineer_features(df_train, df_test):
    """
    Creates new features from existing ones for both training and testing datasets.
    Updates the global FEATURE_COLUMNS list.
    """
    global FEATURE_COLUMNS
    new_features_created = []

    print("\n--- Engineering New Features ---")

    for df_name, df in [("Training", df_train), ("Testing", df_test)]:
        if df is None:
            print(f"Skipping feature engineering for {df_name} data as it's None.")
            continue
        print(f"Engineering features for {df_name} data...")

        # GDP per capita
        if "GDP (current US$)" in df.columns and "Population, total" in df.columns:
            # Replace 0 population with NaN to avoid division by zero, then fill resulting NaNs with 0 or handle later
            df["Population_non_zero"] = df["Population, total"].replace(0, np.nan)
            df["GDP per capita"] = df["GDP (current US$)"] / df["Population_non_zero"]
            df.drop(
                columns=["Population_non_zero"], inplace=True
            )  # clean up temp column
            if "GDP per capita" not in FEATURE_COLUMNS:
                new_features_created.append("GDP per capita")
            print("Created 'GDP per capita'.")
        else:
            print("Could not create 'GDP per capita' due to missing source columns.")

        # Health expenditure per capita
        if (
            "Current health expenditure (% of GDP)" in df.columns
            and "GDP (current US$)" in df.columns
            and "Population, total" in df.columns
        ):
            df["Population_non_zero"] = df["Population, total"].replace(0, np.nan)
            df["Health expenditure per capita"] = (
                df["Current health expenditure (% of GDP)"]
                / 100
                * df["GDP (current US$)"]
            ) / df["Population_non_zero"]
            df.drop(columns=["Population_non_zero"], inplace=True)
            if "Health expenditure per capita" not in FEATURE_COLUMNS:
                new_features_created.append("Health expenditure per capita")
            print("Created 'Health expenditure per capita'.")
        else:
            print(
                "Could not create 'Health expenditure per capita' due to missing source columns."
            )

        # R&D expenditure per capita
        if (
            "Research and development expenditure (% of GDP)" in df.columns
            and "GDP (current US$)" in df.columns
            and "Population, total" in df.columns
        ):
            df["Population_non_zero"] = df["Population, total"].replace(0, np.nan)
            df["R&D expenditure per capita"] = (
                df["Research and development expenditure (% of GDP)"]
                / 100
                * df["GDP (current US$)"]
            ) / df["Population_non_zero"]
            df.drop(columns=["Population_non_zero"], inplace=True)
            if "R&D expenditure per capita" not in FEATURE_COLUMNS:
                new_features_created.append("R&D expenditure per capita")
            print("Created 'R&D expenditure per capita'.")
        else:
            print(
                "Could not create 'R&D expenditure per capita' due to missing source columns."
            )

    # Add unique new features to the global list
    for nf in new_features_created:
        if nf not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append(nf)

    if new_features_created:
        print(f"\nUpdated FEATURE_COLUMNS: {FEATURE_COLUMNS}")
    else:
        print("No new features were added.")
    return df_train, df_test


def data_understanding(df, target_column, feature_columns, data_description="Dataset"):
    """Performs initial data understanding steps on a given DataFrame."""
    if df is None:
        print(
            f"Skipping data understanding for {data_description} as DataFrame is None."
        )
        return False

    print(f"\n--- a. Data Understanding for {data_description} ---")
    print(f"Using features: {feature_columns}")

    required_columns = feature_columns + [target_column]
    missing_essential_cols = [col for col in required_columns if col not in df.columns]
    if missing_essential_cols:
        print(
            f"\nERROR: The following essential columns are missing from the {data_description}: {missing_essential_cols}"
        )
        print(
            "Please check TARGET_COLUMN and FEATURE_COLUMNS configurations and feature engineering steps."
        )
        print(f"Available columns in {data_description} are: {df.columns.tolist()}")
        return False

    print("\n1. Correlation Heatmap:")
    plt.figure(figsize=(14, 12))  # Adjusted size for more features
    try:
        # Ensure all columns for correlation are numeric; handle potential errors by selecting numeric types
        numeric_cols_for_corr = (
            df[feature_columns + [target_column]]
            .select_dtypes(include=np.number)
            .columns
        )
        correlation_matrix = df[numeric_cols_for_corr].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title(f"Correlation Heatmap - {data_description}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate correlation heatmap for {data_description}: {e}")

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

    print("\n3. Missing Data Analysis:")
    columns_to_check_missing = [
        col for col in feature_columns + [target_column] if col in df.columns
    ]
    if not columns_to_check_missing:
        print(
            f"No valid feature or target columns found in {data_description} for missing data analysis."
        )
        return True

    missing_values = df[columns_to_check_missing].isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage (%)": missing_percentage}
    )
    missing_info = missing_info[missing_info["Missing Values"] > 0]

    if not missing_info.empty:
        print(missing_info.sort_values(by="Percentage (%)", ascending=False))
        print(
            "\nNote: Median imputation will be used for missing numerical data during preprocessing."
        )
    else:
        print(
            f"No missing values found in specified features or target column for {data_description}."
        )
    return True


def preprocess_data(X_train, X_test, y_train, y_test):
    """Handles missing data in features and scales features. Handles missing y_train."""
    print("\n--- Preprocessing Data ---")
    # Ensure X_train and X_test are DataFrames for column access
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame")
    if not X_test.empty and not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame if not empty")

    # Impute missing values in features using Median
    # Only impute numeric columns
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns
    numeric_cols_test = (
        X_test.select_dtypes(include=np.number).columns if not X_test.empty else []
    )

    imputer_X = SimpleImputer(strategy="median")

    X_train_imputed_numeric = imputer_X.fit_transform(X_train[numeric_cols_train])
    X_train_imputed_df = pd.DataFrame(
        X_train_imputed_numeric, columns=numeric_cols_train, index=X_train.index
    )
    # For any non-numeric columns, just carry them over (though FEATURE_COLUMNS should be numeric)
    for col in X_train.columns:
        if col not in numeric_cols_train:
            X_train_imputed_df[col] = X_train[col]
    X_train_imputed_df = X_train_imputed_df[X_train.columns]  # Reorder to original

    if not X_test.empty:
        X_test_imputed_numeric = imputer_X.transform(X_test[numeric_cols_test])
        X_test_imputed_df = pd.DataFrame(
            X_test_imputed_numeric, columns=numeric_cols_test, index=X_test.index
        )
        for col in X_test.columns:
            if col not in numeric_cols_test:
                X_test_imputed_df[col] = X_test[col]
        X_test_imputed_df = X_test_imputed_df[X_test.columns]
    else:
        X_test_imputed_df = pd.DataFrame(
            columns=X_train.columns
        )  # Empty df with same columns

    # Scale features
    scaler = StandardScaler()
    # Scale only numeric columns
    X_train_scaled_numeric = scaler.fit_transform(
        X_train_imputed_df[numeric_cols_train]
    )
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled_numeric, columns=numeric_cols_train, index=X_train.index
    )
    for col in X_train_imputed_df.columns:
        if col not in numeric_cols_train:
            X_train_scaled_df[col] = X_train_imputed_df[col]
    X_train_scaled_df = X_train_scaled_df[X_train.columns]

    if not X_test.empty:
        X_test_scaled_numeric = scaler.transform(X_test_imputed_df[numeric_cols_test])
        X_test_scaled_df = pd.DataFrame(
            X_test_scaled_numeric, columns=numeric_cols_test, index=X_test.index
        )
        for col in X_test_imputed_df.columns:
            if col not in numeric_cols_test:
                X_test_scaled_df[col] = X_test_imputed_df[col]
        X_test_scaled_df = X_test_scaled_df[X_test.columns]
    else:
        X_test_scaled_df = pd.DataFrame(columns=X_train.columns)

    y_imputer = SimpleImputer(strategy="median")
    y_train_imputed = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_train_processed = pd.Series(
        y_train_imputed, index=y_train.index, name=y_train.name
    )

    y_test_cleaned = pd.Series(dtype="float64")
    X_test_scaled_df_cleaned = X_test_scaled_df.copy()

    if not y_test.empty:
        y_test_nan_mask = ~pd.Series(y_test).isna()
        y_test_cleaned = y_test[y_test_nan_mask]
        X_test_scaled_df_cleaned = X_test_scaled_df[y_test_nan_mask]

        if not isinstance(y_test_cleaned, pd.Series):
            y_test_cleaned = pd.Series(
                y_test_cleaned,
                index=y_test[y_test_nan_mask].index
                if hasattr(y_test, "index")
                else None,
            )
    print("Preprocessing complete.")
    return (
        X_train_scaled_df,
        X_test_scaled_df_cleaned,
        y_train_processed,
        y_test_cleaned,
    )


def perform_stepwise_selection_and_train(
    X_train, y_train, X_test, y_test, feature_names
):
    """
    Performs Stepwise Forward Selection using mlxtend and trains a Linear Regression model.
    """
    if not MLXTEND_AVAILABLE:
        print("\nSkipping Stepwise Forward Selection as mlxtend is not available.")
        return None, {}

    print(
        "\n--- Performing Stepwise Forward Selection (SFS) with Linear Regression ---"
    )

    if X_train.empty or y_train.empty:
        print("Skipping SFS: Training data is empty.")
        return None, {}

    # Ensure X_train is a DataFrame (should be from preprocessing)
    # and y_train is a Series or 1D array
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train

    lr = LinearRegression()

    sfs = SFS(
        lr,
        k_features=(1, X_train_df.shape[1]),  # Select best number of features
        forward=True,
        floating=False,
        scoring="r2",
        cv=5,  # Use 5-fold cross-validation
        n_jobs=-1,
    )

    print("Fitting SFS model...")
    try:
        sfs = sfs.fit(X_train_df, y_train)
    except Exception as e:
        print(f"Error during SFS fitting: {e}")
        return None, {}

    selected_features_indices = list(sfs.k_feature_idx_)
    selected_features_names = X_train_df.columns[selected_features_indices].tolist()

    if not selected_features_names:
        print("SFS did not select any features. Aborting SFS model training.")
        return None, {}

    print(f"\nSFS selected {len(selected_features_names)} features:")
    for f_name in selected_features_names:
        print(f"- {f_name}")
    print(f"SFS Best R² score during CV: {sfs.k_score_:.4f}")

    X_train_sfs = X_train_df[selected_features_names]
    sfs_model = LinearRegression()
    sfs_model.fit(X_train_sfs, y_train)
    print("Linear Regression model trained with SFS selected features.")

    results_sfs = {}
    if not X_test.empty and not y_test.empty:
        if not isinstance(X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
        else:
            X_test_df = X_test

        X_test_sfs = X_test_df[selected_features_names]
        predictions_sfs = sfs_model.predict(X_test_sfs)
        mse_sfs = mean_squared_error(y_test, predictions_sfs)
        r2_sfs = r2_score(y_test, predictions_sfs)
        results_sfs = {"MSE": mse_sfs, "R²": r2_sfs}
        print(
            f"Linear Regression (SFS) - Test MSE: {mse_sfs:.4f}, Test R²: {r2_sfs:.4f}"
        )

        # Plot feature importances (coefficients) for SFS model
        try:
            sfs_coeffs = pd.Series(
                sfs_model.coef_, index=selected_features_names
            ).sort_values(ascending=False)
            plt.figure(
                figsize=(10, max(6, len(selected_features_names) * 0.5))
            )  # Adjust height
            sfs_coeffs.plot(kind="bar")
            plt.title("Feature Importances (Coefficients) - Linear Regression (SFS)")
            plt.ylabel("Coefficient Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot Linear Regression (SFS) coefficients: {e}")
    else:
        print("Test data is empty, skipping SFS model evaluation on test set.")

    return sfs_model, results_sfs


def modeling_and_evaluation(X_train, y_train, X_test, y_test, feature_columns_list):
    """Trains different models, evaluates them, and identifies important features."""
    if X_train.empty or y_train.empty:
        print("Skipping modeling: Training data (X_train or y_train) is empty.")
        return {}, {}  # Return empty dicts for models and results
    if X_test.empty or y_test.empty:
        print(
            "Skipping modeling evaluation on test set: Test data (X_test or y_test) is empty. "
            "This might be due to all target values in the test set being NaN or test set not loaded."
        )
        # Models can still be trained if X_train/y_train are available.
        # Evaluation on test set will be skipped.

    print("\n--- b. Modeling (on all specified features) ---")

    models = {
        "Linear Regression (All Features)": LinearRegression(),
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

    # Ensure feature_columns_list matches columns in X_train (which is a DataFrame now)
    if isinstance(X_train, pd.DataFrame):
        current_feature_names = X_train.columns.tolist()
    else:  # Should not happen if preprocess_data returns DataFrame
        current_feature_names = feature_columns_list

    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} trained successfully.")

            if not X_test.empty and not y_test.empty:
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results[name] = {"MSE": mse, "R²": r2}
                print(f"{name} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")
            else:
                results[name] = {
                    "MSE": np.nan,
                    "R²": np.nan,
                }  # Indicate no test evaluation
                print(
                    f"{name} - Test evaluation skipped as test data is unavailable/empty."
                )

        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            if name in trained_models:
                del trained_models[name]  # Remove partially trained model
            results[name] = {"MSE": np.nan, "R²": np.nan, "Error": str(e)}

    print("\nFeature Importances (for models trained on all features):")

    lr_model = trained_models.get("Linear Regression (All Features)")
    if lr_model and hasattr(lr_model, "coef_"):
        try:
            lr_coeffs = pd.Series(
                lr_model.coef_, index=current_feature_names
            ).sort_values(ascending=False)
            plt.figure(figsize=(12, max(6, len(current_feature_names) * 0.3)))
            lr_coeffs.plot(kind="bar")
            plt.title(
                "Feature Importances (Coefficients) - Linear Regression (All Features)"
            )
            plt.ylabel("Coefficient Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot Linear Regression (All Features) coefficients: {e}")

    rf_model = trained_models.get("Random Forest Regressor")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        try:
            rf_importances = pd.Series(
                rf_model.feature_importances_, index=current_feature_names
            ).sort_values(ascending=False)
            plt.figure(figsize=(12, max(6, len(current_feature_names) * 0.3)))
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
                gb_model.feature_importances_, index=current_feature_names
            ).sort_values(ascending=False)
            plt.figure(figsize=(12, max(6, len(current_feature_names) * 0.3)))
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
    model,
    X_test,
    y_test,
    model_name,
    target_column_name,
    test_year,
    output_dir="results/task1",
):
    """Visualizes prediction differences and errors for a given model."""
    if model is None:
        print(f"Skipping prediction analysis for {model_name}: Model is None.")
        return
    if X_test.empty or y_test.empty:
        print(f"Skipping prediction analysis for {model_name}: Test data is empty.")
        return
    if not hasattr(model, "predict"):
        print(f"Skipping prediction analysis for {model_name}: Model cannot predict.")
        return

    print(f"\n--- c. Analysis of Predictions for {model_name} (Year {test_year}) ---")
    try:
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction with {model_name}: {e}")
        return

    # Ensure y_test and predictions are 1D arrays/Series
    y_test_np = (
        y_test.values.ravel()
        if isinstance(y_test, pd.Series)
        else np.array(y_test).ravel()
    )
    predictions_np = (
        predictions.ravel()
        if isinstance(predictions, pd.Series)
        else np.array(predictions).ravel()
    )

    if len(y_test_np) != len(predictions_np):
        print(
            f"Length mismatch between y_test ({len(y_test_np)}) and predictions ({len(predictions_np)}). Skipping analysis."
        )
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, predictions_np, alpha=0.6, edgecolors="w", linewidth=0.5)
    min_val = min(y_test_np.min(), predictions_np.min())
    max_val = max(y_test_np.max(), predictions_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    plt.xlabel(f"Actual {target_column_name}")
    plt.ylabel(f"Predicted {target_column_name}")
    plt.title(f"Actual vs. Predicted Life Expectancy ({model_name} - {test_year})")
    plt.grid(True)
    plt.savefig(f"{output_dir}/actual_vs_predicted_{model_name.replace(' ','_')}_{test_year}.png")

    errors = y_test_np - predictions_np
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Prediction Errors ({model_name} - {test_year})")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.savefig(f"{output_dir}/prediction_errors_{model_name.replace(' ','_')}_{test_year}.png")

    errors_series = pd.Series(errors)  # For describe()
    print(f"\nSummary statistics for prediction errors ({model_name}):")
    print(errors_series.describe())

    mean_error = errors_series.mean()
    std_error = errors_series.std()

    if std_error == 0:  # Avoid division by zero or issues with constant errors
        print("Standard deviation of errors is zero. Outlier detection skipped.")
        return

    outlier_threshold_upper = mean_error + 2 * std_error
    outlier_threshold_lower = mean_error - 2 * std_error

    outliers = errors_series[
        (errors_series > outlier_threshold_upper)
        | (errors_series < outlier_threshold_lower)
    ]
    if not outliers.empty:
        print(
            f"\nPotential Outliers (errors > 2 std dev from mean error) for {model_name}:"
        )
        # Assuming y_test has an index that can be used if it's a Series
        outlier_indices = outliers.index
        original_indices = (
            y_test.index[outlier_indices]
            if isinstance(y_test, pd.Series)
            else outlier_indices
        )

        outlier_data = pd.DataFrame(
            {
                "Actual": y_test_np[outlier_indices],
                "Predicted": predictions_np[outlier_indices],
                "Error": outliers.values,  # use .values to align if index is tricky
            },
            index=original_indices,  # Use original index if available
        )
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
    global FEATURE_COLUMNS  # Allow modification by engineer_features

    train_sheet_name = str(YEAR_TRAIN)
    test_sheet_name = str(YEAR_TEST)

    df_train_full, df_test_full = load_data(
        FILE_PATH, train_sheet_name, test_sheet_name
    )

    if df_train_full is None:
        print(
            f"\nExiting: Failed to load training data from sheet '{train_sheet_name}'."
        )
        return

    # --- Feature Engineering ---
    # This will modify df_train_full, df_test_full in place and update global FEATURE_COLUMNS
    df_train_full, df_test_full = engineer_features(df_train_full, df_test_full)

    # --- Data Understanding (on training data with potentially new features) ---
    print(f"\nPerforming data understanding for training data (Year {YEAR_TRAIN})...")
    if not data_understanding(
        df_train_full, TARGET_COLUMN, FEATURE_COLUMNS, f"Training Data ({YEAR_TRAIN})"
    ):
        print("\nExiting due to issues in data understanding for training data.")
        return

    # --- Prepare data for modeling ---
    if TARGET_COLUMN not in df_train_full.columns or not all(
        col in df_train_full.columns for col in FEATURE_COLUMNS
    ):
        print(
            f"Error: Target column '{TARGET_COLUMN}' or some feature columns (current list: {FEATURE_COLUMNS}) not found in training data."
        )
        return

    df_train_full.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df_train_full.empty:
        print(
            f"Error: Training data is empty after dropping rows with NaN in '{TARGET_COLUMN}'."
        )
        return

    X_train = df_train_full[FEATURE_COLUMNS]
    y_train = df_train_full[TARGET_COLUMN]

    X_test = pd.DataFrame()
    y_test = pd.Series(dtype="float64")

    if df_test_full is not None:
        # Check if all current FEATURE_COLUMNS are present in test data
        missing_cols_test = [
            col for col in FEATURE_COLUMNS if col not in df_test_full.columns
        ]
        if TARGET_COLUMN not in df_test_full.columns or missing_cols_test:
            print(
                f"Warning: Target column '{TARGET_COLUMN}' or some feature columns "
                f"(missing: {missing_cols_test if missing_cols_test else 'None'}) "
                f"not found in testing data sheet '{test_sheet_name}'."
            )
            df_test_full = None
        else:
            X_test = df_test_full[FEATURE_COLUMNS]
            y_test = df_test_full[TARGET_COLUMN]
    else:
        print("Test data was not loaded. Evaluation on test set will be skipped.")

    if X_train.empty:
        print(
            f"ERROR: Training features (X_train) are empty after processing sheet '{train_sheet_name}'."
        )
        return

    # --- Preprocessing ---
    # Initialize processed dataframes/series
    X_train_processed = pd.DataFrame()
    X_test_processed = pd.DataFrame()
    y_train_processed = pd.Series(dtype="float64")
    y_test_processed = pd.Series(dtype="float64")

    if df_test_full is None or X_test.empty:
        print("Preprocessing training data only (test data is missing or invalid).")
        # Impute and scale X_train
        numeric_cols_train = X_train.select_dtypes(include=np.number).columns
        imputer_X_train = SimpleImputer(strategy="median")
        X_train_imputed_numeric = imputer_X_train.fit_transform(
            X_train[numeric_cols_train]
        )
        X_train_imputed_df = pd.DataFrame(
            X_train_imputed_numeric, columns=numeric_cols_train, index=X_train.index
        )
        for col in X_train.columns:
            if col not in numeric_cols_train:
                X_train_imputed_df[col] = X_train[col]
        X_train_imputed_df = X_train_imputed_df[X_train.columns]

        scaler_train = StandardScaler()
        X_train_scaled_numeric = scaler_train.fit_transform(
            X_train_imputed_df[numeric_cols_train]
        )
        X_train_processed = pd.DataFrame(
            X_train_scaled_numeric, columns=numeric_cols_train, index=X_train.index
        )
        for col in X_train_imputed_df.columns:
            if col not in numeric_cols_train:
                X_train_processed[col] = X_train_imputed_df[col]
        X_train_processed = X_train_processed[X_train.columns]

        y_imputer_train = SimpleImputer(strategy="median")
        y_train_imputed_arr = y_imputer_train.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()
        y_train_processed = pd.Series(
            y_train_imputed_arr, index=y_train.index, name=y_train.name
        )

        # X_test_processed and y_test_processed remain empty
        X_test_processed = pd.DataFrame(
            columns=X_train_processed.columns
        )  # Empty with correct columns
        y_test_processed = pd.Series(dtype="float64")

    else:  # Both train and test data seem okay
        print("\nPreprocessing training and testing data (Imputation and Scaling)...")
        X_train_processed, X_test_processed, y_train_processed, y_test_processed = (
            preprocess_data(X_train, X_test, y_train, y_test)
        )

    if X_train_processed.empty or y_train_processed.empty:
        print("Training data is empty after preprocessing. Cannot proceed.")
        return

    all_trained_models = {}
    all_results = {}

    # --- Stepwise Forward Selection (SFS) ---
    # The feature names for SFS come from the columns of X_train_processed
    sfs_model, sfs_results = perform_stepwise_selection_and_train(
        X_train_processed.copy(),  # Pass copy to avoid modification issues
        y_train_processed.copy(),
        X_test_processed.copy() if not X_test_processed.empty else pd.DataFrame(),
        y_test_processed.copy()
        if not y_test_processed.empty
        else pd.Series(dtype="float64"),
        X_train_processed.columns.tolist(),  # Pass current feature names
    )
    if sfs_model:
        all_trained_models["Linear Regression (SFS)"] = sfs_model
    if sfs_results:
        all_results["Linear Regression (SFS)"] = sfs_results

    # --- Modeling and Evaluation (on all features) ---
    # FEATURE_COLUMNS here should be the full list (original + engineered)
    # which are the columns of X_train_processed
    trained_models_all_feat, results_all_feat = modeling_and_evaluation(
        X_train_processed,
        y_train_processed,
        X_test_processed if not X_test_processed.empty else pd.DataFrame(),
        y_test_processed if not y_test_processed.empty else pd.Series(dtype="float64"),
        X_train_processed.columns.tolist(),  # Pass current feature names
    )
    all_trained_models.update(trained_models_all_feat)
    all_results.update(results_all_feat)

    print("\n--- Combined Model Performance Summary ---")
    if all_results:
        for name, metrics in all_results.items():
            if "Error" in metrics:
                print(f"{name} - Error: {metrics['Error']}")
            elif np.isnan(metrics.get("MSE", np.nan)):
                print(f"{name} - Trained, but no test evaluation.")
            else:
                print(
                    f"{name} - Test MSE: {metrics.get('MSE', np.nan):.4f}, Test R²: {metrics.get('R²', np.nan):.4f}"
                )
    else:
        print("No models were successfully evaluated.")

    if all_trained_models and not X_test_processed.empty and not y_test_processed.empty:
        best_model_name = None
        best_r2 = -float("inf")

        if all_results:
            for name, metrics in all_results.items():
                # Only consider models with valid R² scores for "best"
                if pd.notna(metrics.get("R²")) and metrics["R²"] > best_r2:
                    best_r2 = metrics["R²"]
                    best_model_name = name

        model_to_analyze_name = None
        if best_model_name and best_model_name in all_trained_models:
            model_to_analyze_name = best_model_name
            print(
                f"\nAnalyzing predictions for the best model based on R² on Test Set: {model_to_analyze_name} (R²: {best_r2:.4f})"
            )
        elif (
            "Random Forest Regressor" in all_trained_models
        ):  # Fallback if R2 based selection fails
            model_to_analyze_name = "Random Forest Regressor"
            print(
                f"\nCould not determine best model by R² or no test results. Defaulting analysis to Random Forest Regressor."
            )
        elif all_trained_models:  # Fallback to any available model
            model_to_analyze_name = list(all_trained_models.keys())[0]
            print(
                f"\nCould not determine best model. Analyzing first available model: {model_to_analyze_name}"
            )

        if model_to_analyze_name and model_to_analyze_name in all_trained_models:
            # For SFS model, X_test needs to be subsetted to selected features
            current_X_test_for_analysis = X_test_processed
            if (
                model_to_analyze_name == "Linear Regression (SFS)"
                and sfs_model is not None
            ):
                # Need the selected feature names from SFS step
                sfs_selected_cols = [
                    col
                    for col in X_train_processed.columns
                    if col
                    in all_trained_models[model_to_analyze_name].feature_names_in_
                ]  # Get selected features
                if sfs_selected_cols:
                    current_X_test_for_analysis = X_test_processed[sfs_selected_cols]
                else:  # Should not happen if SFS model is valid
                    print(
                        f"Warning: Could not get SFS selected features for {model_to_analyze_name}. Analysis might be incorrect."
                    )

            analysis_of_predictions(
                all_trained_models[model_to_analyze_name],
                current_X_test_for_analysis,  # Use potentially subsetted X_test for SFS
                y_test_processed,
                model_to_analyze_name,
                TARGET_COLUMN,
                YEAR_TEST,
            )
        else:
            print(
                "\nNo suitable model available for detailed prediction analysis on test data."
            )

    elif all_trained_models:
        print(
            "\nModels were trained, but test data was not available or suitable for evaluation/prediction analysis."
        )
    else:
        print(
            "\nModeling step did not produce any models. Skipping prediction analysis."
        )

    print("\n--- End of Analysis ---")


if __name__ == "__main__":
    # Create a dummy Excel file for testing if it doesn't exist
    # This is just for basic script execution without the actual file.
    # Replace with your actual file path.
    try:
        pd.read_excel(
            FILE_PATH, sheet_name=None
        )  # Check if file exists and is readable
    except FileNotFoundError:
        print(
            f"Warning: '{FILE_PATH}' not found. Creating a dummy file for script execution."
        )
        print("Please replace with your actual data file.")
        dummy_cols = [
            "Country Name",
            "Life expectancy at birth, total (years)",
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
        dummy_data_2008 = pd.DataFrame(
            np.random.rand(20, len(dummy_cols)), columns=dummy_cols
        )
        dummy_data_2008["Country Name"] = [f"Country_{i}" for i in range(20)]
        dummy_data_2008["GDP (current US$)"] = np.random.randint(1e9, 1e12, 20)
        dummy_data_2008["Population, total"] = np.random.randint(1e6, 1e8, 20)
        dummy_data_2008["Life expectancy at birth, total (years)"] = np.random.uniform(
            50, 85, 20
        )

        dummy_data_2018 = pd.DataFrame(
            np.random.rand(20, len(dummy_cols)), columns=dummy_cols
        )
        dummy_data_2018["Country Name"] = [f"Country_{i}" for i in range(20)]
        dummy_data_2018["GDP (current US$)"] = np.random.randint(1e9, 1e12, 20)
        dummy_data_2018["Population, total"] = np.random.randint(1e6, 1e8, 20)
        dummy_data_2018["Life expectancy at birth, total (years)"] = np.random.uniform(
            50, 85, 20
        )

        with pd.ExcelWriter(FILE_PATH) as writer:
            dummy_data_2008.to_excel(writer, sheet_name=str(YEAR_TRAIN), index=False)
            dummy_data_2018.to_excel(writer, sheet_name=str(YEAR_TEST), index=False)
        print(
            f"Dummy file '{FILE_PATH}' created with sheets for {YEAR_TRAIN} and {YEAR_TEST}."
        )

    main()
    if plt.get_fignums():
        print("\nDisplaying plots. Close plot windows to end script.")
        plt.show(block=True)
    else:
        print("\nNo plots were generated to display.")

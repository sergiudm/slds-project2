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
import os  # For creating output directory

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

FILE_PATH = "assets/life_indicator_2008-2018.xlsx"
YEAR_TRAIN_START = 2008
YEAR_TRAIN_END = 2017  # Train up to 2017
YEAR_TEST = 2018  # Test on 2018, also use 2018 features as proxy for 2025
YEAR_PREDICT_FUTURE = 2025  # The future year we want to predict for
OUTPUT_DIR = "results/task1_multiyear_before"
FUTURE_PREDICTION_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR, f"{YEAR_PREDICT_FUTURE}_predictions"
)


def load_all_relevant_data(file_path, start_year, end_year, test_year):
    """Loads data from specified sheets (years) in the Excel file."""
    all_sheets_data = {}
    years_to_load = [str(y) for y in range(start_year, end_year + 1)] + [str(test_year)]
    print(f"Attempting to load data for years: {', '.join(years_to_load)}")

    try:
        excel_file = pd.ExcelFile(file_path)
        available_sheets = excel_file.sheet_names
        print(f"Available sheets in '{file_path}': {available_sheets}")

        for year_str in years_to_load:
            if year_str in available_sheets:
                print(f"Loading data for year: '{year_str}'")
                try:
                    df = excel_file.parse(year_str)
                    all_sheets_data[year_str] = df
                    print(
                        f"Data for year '{year_str}' loaded successfully. Shape: {df.shape}"
                    )
                except Exception as e:
                    print(f"Error loading sheet '{year_str}': {e}")
                    all_sheets_data[year_str] = None
            else:
                print(
                    f"Warning: Worksheet named '{year_str}' not found in '{file_path}'."
                )
                all_sheets_data[year_str] = None

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while trying to open or read '{file_path}': {e}")
        return None
    return all_sheets_data


def engineer_features(df_train, df_test):
    """
    Creates new features from existing ones for both training and testing datasets.
    Updates the global FEATURE_COLUMNS list.
    """
    global FEATURE_COLUMNS
    new_features_created_local = []

    print("\n--- Engineering New Features ---")

    dataframes_to_process = []
    if df_train is not None and not df_train.empty:
        dataframes_to_process.append(("Training", df_train))
    if df_test is not None and not df_test.empty:
        dataframes_to_process.append(("Testing", df_test))

    if not dataframes_to_process:
        print("No dataframes provided for feature engineering.")
        return df_train, df_test

    for df_name, df in dataframes_to_process:
        print(f"Engineering features for {df_name} data...")
        if df is None or df.empty:
            print(f"Skipping feature engineering for {df_name} as it's empty.")
            continue

        # GDP per capita
        if "GDP (current US$)" in df.columns and "Population, total" in df.columns:
            df["Population_non_zero"] = df["Population, total"].replace(0, np.nan)
            df["GDP per capita"] = df["GDP (current US$)"] / df["Population_non_zero"]
            df.drop(columns=["Population_non_zero"], inplace=True)
            if "GDP per capita" not in new_features_created_local:
                new_features_created_local.append("GDP per capita")
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
            if "Health expenditure per capita" not in new_features_created_local:
                new_features_created_local.append("Health expenditure per capita")
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
            if "R&D expenditure per capita" not in new_features_created_local:
                new_features_created_local.append("R&D expenditure per capita")
            print("Created 'R&D expenditure per capita'.")
        else:
            print(
                "Could not create 'R&D expenditure per capita' due to missing source columns."
            )

    for nf in new_features_created_local:
        if nf not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append(nf)

    if new_features_created_local:
        print(f"\nUpdated FEATURE_COLUMNS globally: {FEATURE_COLUMNS}")
    else:
        print("No new features were added in this call to engineer_features.")
    return df_train, df_test


def data_understanding(
    df, target_column, feature_columns, data_description="Dataset", output_dir="."
):
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

    os.makedirs(output_dir, exist_ok=True)

    print("\n1. Correlation Heatmap:")
    plt.figure(figsize=(18, 15))  # Adjusted size
    try:
        numeric_cols_for_corr = (
            df[feature_columns + [target_column]]
            .select_dtypes(include=np.number)
            .columns
        )
        if not numeric_cols_for_corr.empty:
            correlation_matrix = df[numeric_cols_for_corr].corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                annot_kws={"size": 7},  # Smaller font for more features
            )
            plt.title(f"Correlation Heatmap - {data_description}")
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/correlation_heatmap_{data_description.replace(' ', '_').lower()}.png"
            )
            plt.close()  # Close plot after saving
        else:
            print("No numeric columns found for correlation heatmap.")
    except Exception as e:
        print(f"Could not generate correlation heatmap for {data_description}: {e}")

    print(f"\n2. Distribution of '{target_column}':")
    if target_column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_column].dropna(), kde=True)
        plt.title(f"Distribution of {target_column} - {data_description}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.savefig(
            f"{output_dir}/target_distribution_{data_description.replace(' ', '_').lower()}.png"
        )
        plt.close()  # Close plot after saving
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
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame")
    if X_test is not None and not X_test.empty and not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame if not empty")

    numeric_cols_train = X_train.select_dtypes(include=np.number).columns
    imputer_X = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # --- Process Training Data ---
    X_train_imputed_numeric = imputer_X.fit_transform(X_train[numeric_cols_train])
    X_train_imputed_df = pd.DataFrame(
        X_train_imputed_numeric, columns=numeric_cols_train, index=X_train.index
    )
    # Add back non-numeric columns if any (should be rare for these features)
    for col in X_train.columns:
        if col not in numeric_cols_train:
            X_train_imputed_df[col] = X_train[col]
    X_train_imputed_df = X_train_imputed_df[X_train.columns]  # Reorder

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

    y_imputer = SimpleImputer(strategy="median")
    y_train_imputed = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_train_processed = pd.Series(
        y_train_imputed, index=y_train.index, name=y_train.name
    )

    # --- Process Test Data (if exists) ---
    X_test_scaled_df_cleaned = pd.DataFrame(columns=X_train.columns)  # Default empty
    y_test_cleaned = pd.Series(dtype="float64")

    if X_test is not None and not X_test.empty:
        # Ensure X_test has all columns from X_train before imputation/scaling
        for col in X_train.columns:
            if col not in X_test.columns:
                print(
                    f"Warning: Column '{col}' from training data not in test data. Adding with NaNs."
                )
                X_test[col] = np.nan
        X_test = X_test[X_train.columns]  # Order and subset X_test like X_train

        numeric_cols_test = X_test.select_dtypes(include=np.number).columns
        X_test_imputed_numeric = imputer_X.transform(X_test[numeric_cols_test])
        X_test_imputed_df = pd.DataFrame(
            X_test_imputed_numeric, columns=numeric_cols_test, index=X_test.index
        )
        for col in X_test.columns:
            if col not in numeric_cols_test:
                X_test_imputed_df[col] = X_test[col]
        X_test_imputed_df = X_test_imputed_df[X_test.columns]

        X_test_scaled_numeric = scaler.transform(X_test_imputed_df[numeric_cols_test])
        X_test_scaled_df = pd.DataFrame(
            X_test_scaled_numeric, columns=numeric_cols_test, index=X_test.index
        )
        for col in X_test_imputed_df.columns:
            if col not in numeric_cols_test:
                X_test_scaled_df[col] = X_test_imputed_df[col]
        X_test_scaled_df = X_test_scaled_df[X_test.columns]

        # Clean y_test and align X_test_scaled_df
        if y_test is not None and not y_test.empty:
            y_test_nan_mask = ~pd.Series(
                y_test
            ).isna()  # Ensure y_test is a Series for .isna()
            y_test_cleaned = y_test[y_test_nan_mask]
            X_test_scaled_df_cleaned = X_test_scaled_df[y_test_nan_mask]
            if not isinstance(y_test_cleaned, pd.Series):  # Re-ensure Series type
                y_test_cleaned = pd.Series(
                    y_test_cleaned,
                    index=y_test[y_test_nan_mask].index,
                    name=y_test.name,
                )
        else:  # If y_test is empty to begin with
            X_test_scaled_df_cleaned = (
                X_test_scaled_df  # Use all X_test rows if no y_test to filter by
            )
    else:  # If X_test itself is None or empty
        X_test_scaled_df_cleaned = pd.DataFrame(
            columns=X_train.columns
        )  # Empty df with train columns
        y_test_cleaned = pd.Series(dtype="float64")

    print("Preprocessing complete.")
    # Return the scaler and imputer_X as they might be needed for future data (like 2025 predictions on raw features)
    # However, for this specific request, we'll use X_test_scaled_df_cleaned directly as proxy for 2025 processed features
    return (
        X_train_scaled_df,
        X_test_scaled_df_cleaned,  # This is X_test processed and cleaned based on y_test NaNs
        y_train_processed,
        y_test_cleaned,
        X_test_scaled_df,  # Return the full processed X_test before y_test NaN cleaning for future predictions
    )


def perform_stepwise_selection_and_train(
    X_train, y_train, X_test, y_test, feature_names, output_dir="."
):
    """
    Performs Stepwise Forward Selection using mlxtend and trains a Linear Regression model.
    Returns the model, its test results, and the list of selected feature names.
    """
    selected_features_names = None  # Initialize
    if not MLXTEND_AVAILABLE:
        print("\nSkipping Stepwise Forward Selection as mlxtend is not available.")
        return None, {}, None

    print(
        "\n--- Performing Stepwise Forward Selection (SFS) with Linear Regression ---"
    )
    os.makedirs(output_dir, exist_ok=True)

    if X_train.empty or y_train.empty:
        print("Skipping SFS: Training data is empty.")
        return None, {}, None

    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train

    lr = LinearRegression()
    # Adjust cv dynamically
    n_samples_train = X_train_df.shape[0]
    cv_folds = (
        min(5, n_samples_train) if n_samples_train > 1 else 0
    )  # Use 0 for LOO if samples < k_folds
    if cv_folds <= 1 and n_samples_train > 1:  # Must be at least 2 for KFold
        cv_folds = max(2, n_samples_train - 1) if n_samples_train > 1 else 0

    sfs = SFS(
        lr,
        k_features=(1, X_train_df.shape[1]),  # Max features can be all features
        forward=True,
        floating=False,
        scoring="r2",
        cv=cv_folds,
        n_jobs=-1,
    )

    print(
        f"Fitting SFS model... (CV folds: {cv_folds if cv_folds > 0 else 'LOO (if cv=0 and samples >1)'})"
    )
    try:
        if cv_folds == 0 and n_samples_train <= 1:  # Cannot fit if only 1 sample
            print("Cannot perform SFS with 1 or 0 samples. Skipping SFS.")
            return None, {}, None
        sfs = sfs.fit(
            X_train_df.to_numpy(), y_train.to_numpy()
        )  # Use .to_numpy() for recent scikit-learn/mlxtend
    except ValueError as ve:
        print(f"Error during SFS fitting: {ve}. Trying with reduced CV if applicable.")
        if "n_splits=" in str(
            ve
        ) and "cannot be greater than the number of samples" in str(ve):
            try:
                sfs.cv = max(2, n_samples_train - 1) if n_samples_train > 1 else 0
                if sfs.cv >= 2:  # mlxtend SFS cv must be >=2 or 0 (LOO)
                    sfs = sfs.fit(X_train_df.to_numpy(), y_train.to_numpy())
                else:
                    print("Cannot perform SFS with CV < 2. Skipping SFS.")
                    return None, {}, None
            except Exception as e_inner:
                print(f"Error during SFS fitting (retry): {e_inner}")
                return None, {}, None
        else:
            print(f"Unhandled error during SFS fitting: {ve}")
            return None, {}, None
    except Exception as e:
        print(f"Generic error during SFS fitting: {e}")
        return None, {}, None

    selected_features_indices = list(sfs.k_feature_idx_)
    if not selected_features_indices:
        print("SFS did not select any features. Aborting SFS model training.")
        return None, {}, None

    selected_features_names = X_train_df.columns[selected_features_indices].tolist()

    if not selected_features_names:
        print(
            "SFS did not select any features (names list empty). Aborting SFS model training."
        )
        return None, {}, None

    print(f"\nSFS selected {len(selected_features_names)} features:")
    for f_name in selected_features_names:
        print(f"- {f_name}")
    print(f"SFS Best R² score during CV: {sfs.k_score_:.4f}")

    X_train_sfs = X_train_df[selected_features_names]
    sfs_model = LinearRegression()
    sfs_model.fit(X_train_sfs, y_train)
    print("Linear Regression model trained with SFS selected features.")

    results_sfs = {}
    if (
        X_test is not None
        and not X_test.empty
        and y_test is not None
        and not y_test.empty
    ):
        if not isinstance(X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
        else:
            X_test_df = X_test

        missing_sfs_cols_in_test = [
            col for col in selected_features_names if col not in X_test_df.columns
        ]
        if missing_sfs_cols_in_test:
            print(
                f"Warning: SFS selected features missing from test data: {missing_sfs_cols_in_test}. Cannot evaluate SFS model on test set."
            )
            return (
                sfs_model,
                {
                    "MSE": np.nan,
                    "R²": np.nan,
                    "Warning": "Missing SFS features in test data",
                },
                selected_features_names,
            )

        X_test_sfs = X_test_df[selected_features_names]
        predictions_sfs = sfs_model.predict(X_test_sfs)

        if len(y_test) == len(predictions_sfs):
            mse_sfs = mean_squared_error(y_test, predictions_sfs)
            r2_sfs = r2_score(y_test, predictions_sfs)
            results_sfs = {"MSE": mse_sfs, "R²": r2_sfs}
            print(
                f"Linear Regression (SFS) - Test MSE: {mse_sfs:.4f}, Test R²: {r2_sfs:.4f}"
            )
        else:
            print(
                f"Length mismatch between y_test ({len(y_test)}) and SFS predictions ({len(predictions_sfs)}). Skipping SFS evaluation."
            )
            results_sfs = {
                "MSE": np.nan,
                "R²": np.nan,
                "Warning": "Length mismatch in SFS evaluation",
            }

        try:
            sfs_coeffs = pd.Series(
                sfs_model.coef_, index=selected_features_names
            ).sort_values(ascending=False)
            plt.figure(figsize=(10, max(6, len(selected_features_names) * 0.5)))
            sfs_coeffs.plot(kind="bar")
            plt.title("Feature Importances (Coefficients) - Linear Regression (SFS)")
            plt.ylabel("Coefficient Value")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sfs_feature_importances.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot Linear Regression (SFS) coefficients: {e}")
    else:
        print(
            "Test data is empty or unavailable, skipping SFS model evaluation on test set."
        )
        results_sfs = {"MSE": np.nan, "R²": np.nan}

    return sfs_model, results_sfs, selected_features_names


def modeling_and_evaluation(
    X_train, y_train, X_test, y_test, feature_columns_list, output_dir="."
):
    """Trains different models, evaluates them, and identifies important features."""
    if X_train.empty or y_train.empty:
        print("Skipping modeling: Training data (X_train or y_train) is empty.")
        return {}, {}

    os.makedirs(output_dir, exist_ok=True)
    perform_test_evaluation = True
    if X_test is None or X_test.empty or y_test is None or y_test.empty:
        print(
            "Skipping modeling evaluation on test set: Test data is empty or unavailable."
        )
        perform_test_evaluation = False

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
    current_feature_names = (
        X_train.columns.tolist()
        if isinstance(X_train, pd.DataFrame)
        else feature_columns_list
    )

    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} trained successfully.")

            if perform_test_evaluation:
                missing_cols_in_test = [
                    col for col in current_feature_names if col not in X_test.columns
                ]
                if missing_cols_in_test:
                    print(
                        f"Warning for {name}: Test data missing columns: {missing_cols_in_test}. Cannot evaluate."
                    )
                    results[name] = {
                        "MSE": np.nan,
                        "R²": np.nan,
                        "Error": "Test data missing columns",
                    }
                    continue

                X_test_ordered = X_test[current_feature_names]
                predictions = model.predict(X_test_ordered)

                if len(y_test) == len(predictions):
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    results[name] = {"MSE": mse, "R²": r2}
                    print(f"{name} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")
                else:
                    print(
                        f"Length mismatch for {name}: y_test ({len(y_test)}), predictions ({len(predictions)}). Skipping eval."
                    )
                    results[name] = {
                        "MSE": np.nan,
                        "R²": np.nan,
                        "Warning": "Length mismatch in evaluation",
                    }
            else:
                results[name] = {"MSE": np.nan, "R²": np.nan}
                print(f"{name} - Test evaluation skipped.")
        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            if name in trained_models:
                del trained_models[name]
            results[name] = {"MSE": np.nan, "R²": np.nan, "Error": str(e)}

    print("\nFeature Importances (for models trained on all features):")
    # ... (plotting code for LR, RF, GB - ensure plt.close() is used)
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
            plt.savefig(f"{output_dir}/lr_all_features_importances.png")
            plt.close()
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
            plt.savefig(f"{output_dir}/rf_feature_importances.png")
            plt.close()
            # ... (print determination method)
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
            plt.savefig(f"{output_dir}/gb_feature_importances.png")
            plt.close()
            # ... (print determination method)
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
    output_dir=".",
    feature_names_for_model=None,
):
    """Visualizes prediction differences and errors for a given model."""
    if (
        model is None
        or X_test is None
        or X_test.empty
        or y_test is None
        or y_test.empty
        or not hasattr(model, "predict")
    ):
        print(f"Skipping prediction analysis for {model_name}: Invalid inputs.")
        return

    print(f"\n--- c. Analysis of Predictions for {model_name} (Year {test_year}) ---")
    os.makedirs(output_dir, exist_ok=True)

    X_test_for_prediction = X_test
    if feature_names_for_model:
        missing_model_cols = [
            col for col in feature_names_for_model if col not in X_test.columns
        ]
        if missing_model_cols:
            print(
                f"Warning for {model_name} analysis: Test data missing model-specific columns: {missing_model_cols}."
            )
            return
        X_test_for_prediction = X_test[feature_names_for_model]

    try:
        predictions = model.predict(X_test_for_prediction)
    except Exception as e:
        print(f"Error during prediction with {model_name}: {e}")
        return

    if len(y_test) != len(predictions):
        print(
            f"Length mismatch for {model_name} analysis: y_test ({len(y_test)}), predictions ({len(predictions)}). Skipping."
        )
        return

    y_test_np = (
        y_test.values.ravel()
        if isinstance(y_test, pd.Series)
        else np.array(y_test).ravel()
    )
    # ... (rest of the plotting and analysis code, ensure plt.close() after saving figs)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, predictions, alpha=0.6, edgecolors="w", linewidth=0.5)
    min_val = (
        min(y_test_np.min(), predictions.min())
        if len(y_test_np) > 0 and len(predictions) > 0
        else 0
    )
    max_val = (
        max(y_test_np.max(), predictions.max())
        if len(y_test_np) > 0 and len(predictions) > 0
        else 1
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    plt.xlabel(f"Actual {target_column_name}")
    plt.ylabel(f"Predicted {target_column_name}")
    plt.title(f"Actual vs. Predicted ({model_name} - {test_year})")
    plt.grid(True)
    plt.savefig(
        f"{output_dir}/actual_vs_predicted_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{test_year}.png"
    )
    plt.close()

    errors = y_test_np - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Prediction Errors ({model_name} - {test_year})")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.savefig(
        f"{output_dir}/prediction_errors_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{test_year}.png"
    )
    plt.close()
    # ... (outlier analysis)


def predict_for_future_year(
    model,
    X_future_scenario_features,  # These are the processed features for the future scenario (e.g., from YEAR_TEST)
    df_base_year_for_context,  # Original dataframe for the base year (e.g., YEAR_TEST) to get country names
    model_name,
    target_column_name,
    future_year,
    output_dir,
    country_column_name,
    feature_names_for_model=None,  # Specific features if model (e.g. SFS) uses a subset
    base_year_actual_target=None,  # Optional: y_test (actuals from base year) for context
):
    """
    Predicts target for a future year using a trained model and scenario features.
    Visualizes and saves the predictions.
    Assumes X_future_scenario_features and df_base_year_for_context are aligned if country names are to be used.
    """
    if model is None:
        print(f"Skipping {future_year} prediction for {model_name}: Model is None.")
        return None
    if X_future_scenario_features is None or X_future_scenario_features.empty:
        print(
            f"Skipping {future_year} prediction for {model_name}: Input features are empty."
        )
        return None

    print(
        f"\n--- Predicting for {future_year} using {model_name} (based on {YEAR_TEST} indicators) ---"
    )
    os.makedirs(output_dir, exist_ok=True)

    X_predict = X_future_scenario_features
    if feature_names_for_model:
        print(f"Using specific features for {model_name}: {feature_names_for_model}")
        missing_cols = [
            col for col in feature_names_for_model if col not in X_predict.columns
        ]
        if missing_cols:
            print(
                f"Error: Features {missing_cols} required by {model_name} are missing from input data. Cannot predict."
            )
            return None
        X_predict = X_predict[feature_names_for_model]

    try:
        future_predictions = model.predict(X_predict)
    except Exception as e:
        print(f"Error during {future_year} prediction with {model_name}: {e}")
        return None

    predictions_df = pd.DataFrame()
    # Align predictions with country names
    # Ensure indices match between X_future_scenario_features and df_base_year_for_context
    if country_column_name in df_base_year_for_context.columns and len(
        future_predictions
    ) == len(df_base_year_for_context):
        # If df_base_year_for_context was filtered, its index might be non-contiguous.
        # X_future_scenario_features should have a matching index.
        predictions_df[country_column_name] = df_base_year_for_context[
            country_column_name
        ].values
        predictions_df[f"Predicted {target_column_name} ({future_year})"] = (
            future_predictions
        )

        if base_year_actual_target is not None and len(base_year_actual_target) == len(
            future_predictions
        ):
            predictions_df[f"Actual {target_column_name} ({YEAR_TEST}) for Context"] = (
                base_year_actual_target.values
            )
    else:
        print(
            "Warning: Could not align predictions with country names or base year actuals due to mismatched lengths or missing country column."
        )
        print(
            f"Length of predictions: {len(future_predictions)}, Length of df_base_year: {len(df_base_year_for_context)}"
        )
        predictions_df[f"Predicted {target_column_name} ({future_year})"] = (
            future_predictions
        )

    print(f"\n{future_year} Predictions Summary ({model_name}):")
    print(predictions_df.head())
    predictions_df.to_csv(
        f"{output_dir}/{model_name.replace(' ', '_')}_{future_year}_predictions.csv",
        index=False,
    )
    print(f"Saved {future_year} predictions to CSV.")

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(
        predictions_df[f"Predicted {target_column_name} ({future_year})"], kde=True
    )
    plt.title(
        f"Distribution of Predicted {target_column_name} for {future_year}\n(Model: {model_name}, Based on {YEAR_TEST} Indicators)"
    )
    plt.xlabel(f"Predicted {target_column_name}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(
        f"{output_dir}/predicted_{future_year}_distribution_{model_name.replace(' ', '_')}.png"
    )
    plt.close()
    print(f"Saved {future_year} prediction distribution plot.")

    # Bar chart for a sample of countries if country names are available
    if country_column_name in predictions_df.columns and not predictions_df.empty:
        sample_size = min(20, len(predictions_df))
        sample_predictions = predictions_df.sample(
            sample_size, random_state=42
        ).sort_values(
            by=f"Predicted {target_column_name} ({future_year})", ascending=False
        )
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x=f"Predicted {target_column_name} ({future_year})",
            y=country_column_name,
            data=sample_predictions,
            palette="viridis",
        )
        plt.title(
            f"Sample Predicted {target_column_name} for {future_year} by Country\n(Model: {model_name})"
        )
        plt.xlabel(f"Predicted {target_column_name}")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/predicted_{future_year}_sample_countries_{model_name.replace(' ', '_')}.png"
        )
        plt.close()
        print(f"Saved {future_year} sample country predictions plot.")

    return predictions_df


def main():
    global FEATURE_COLUMNS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FUTURE_PREDICTION_OUTPUT_DIR, exist_ok=True)

    all_loaded_data = load_all_relevant_data(
        FILE_PATH, YEAR_TRAIN_START, YEAR_TRAIN_END, YEAR_TEST
    )
    if all_loaded_data is None:
        print("\nExiting: Failed to load any data.")
        return

    df_train_list = []
    print(
        f"\n--- Aggregating Training Data from {YEAR_TRAIN_START} to {YEAR_TRAIN_END} ---"
    )
    for year in range(YEAR_TRAIN_START, YEAR_TRAIN_END + 1):
        year_str = str(year)
        if year_str in all_loaded_data and all_loaded_data[year_str] is not None:
            print(f"Adding data from year {year_str} to training set.")
            df_year = all_loaded_data[year_str].copy()
            if "Year" not in df_year.columns:
                df_year["Year"] = year
            df_train_list.append(df_year)
        else:
            print(f"Warning: Data for training year {year_str} is missing. Skipping.")

    if not df_train_list:
        print("\nExiting: No training data could be aggregated.")
        return
    df_train_full = pd.concat(df_train_list, ignore_index=True)
    print(f"Combined training data shape: {df_train_full.shape}")

    df_test_full = all_loaded_data.get(str(YEAR_TEST))
    if df_test_full is not None:
        df_test_full = df_test_full.copy()  # Ensure it's a copy
        if "Year" not in df_test_full.columns:
            df_test_full["Year"] = YEAR_TEST
        print(f"Test data ({YEAR_TEST}) shape: {df_test_full.shape}")
    else:
        print(
            f"Warning: Test data for year {YEAR_TEST} could not be loaded. Test set eval & future prediction will be limited."
        )

    # --- Feature Engineering ---
    # Make copies before passing to engineer_features
    df_train_full_fe = df_train_full.copy() if df_train_full is not None else None
    df_test_full_fe = df_test_full.copy() if df_test_full is not None else None

    df_train_full_fe, df_test_full_fe = engineer_features(
        df_train_full_fe, df_test_full_fe
    )  # UNCOMMENTED

    print(
        f"\nPerforming data understanding for combined training data ({YEAR_TRAIN_START}-{YEAR_TRAIN_END})..."
    )
    if df_train_full_fe is not None and not data_understanding(
        df_train_full_fe,
        TARGET_COLUMN,
        FEATURE_COLUMNS,
        f"Training Data ({YEAR_TRAIN_START}-{YEAR_TRAIN_END})",
        output_dir=OUTPUT_DIR,
    ):
        print("\nExiting due to issues in data understanding for training data.")
        return
    elif df_train_full_fe is None:
        print("Skipping data understanding as combined training data is None.")
        return

    if TARGET_COLUMN not in df_train_full_fe.columns or not all(
        col in df_train_full_fe.columns for col in FEATURE_COLUMNS
    ):
        print(
            f"Error: Target or some features not in combined training data. Available: {df_train_full_fe.columns.tolist()}"
        )
        return

    df_train_full_fe.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df_train_full_fe.empty:
        print(f"Error: Training data empty after dropping NaN targets.")
        return

    X_train = df_train_full_fe[FEATURE_COLUMNS]
    y_train = df_train_full_fe[TARGET_COLUMN]

    X_test = pd.DataFrame()
    y_test = pd.Series(dtype="float64")
    df_test_for_context = (
        pd.DataFrame()
    )  # To carry country names aligned with X_test_processed/y_test_processed

    if df_test_full_fe is not None:
        # Check for all original + engineered features in test data
        missing_cols_test_fe = [
            col for col in FEATURE_COLUMNS if col not in df_test_full_fe.columns
        ]
        if TARGET_COLUMN not in df_test_full_fe.columns or missing_cols_test_fe:
            print(
                f"Warning: Target ('{TARGET_COLUMN}') or some features (missing: {missing_cols_test_fe}) "
                f"not found in testing data for {YEAR_TEST} after FE. Eval/Future pred may be affected."
            )
            # X_test, y_test will remain empty
        else:
            # Filter test data where target is NaN for consistent evaluation
            # This mask is important for aligning data for future predictions context
            y_test_nan_mask = df_test_full_fe[TARGET_COLUMN].notna()

            df_test_analysis_ready = df_test_full_fe[
                y_test_nan_mask
            ].copy()  # Used for analysis & future prediction context
            X_test = df_test_analysis_ready[FEATURE_COLUMNS]
            y_test = df_test_analysis_ready[TARGET_COLUMN]
            df_test_for_context = df_test_analysis_ready[
                [COUNTRY_COLUMN, TARGET_COLUMN]
            ].reset_index(drop=True)

            # X_test_for_future_prediction_base will be derived from pre-processing X_test without y-based NaN removal
            # This means we use ALL countries from YEAR_TEST for future prediction if they have features
            X_test_for_future_base_raw = df_test_full_fe[FEATURE_COLUMNS].copy()

    else:
        print(
            f"Test data for {YEAR_TEST} was not loaded/is empty. Eval & Future pred will be skipped/limited."
        )
        X_test_for_future_base_raw = pd.DataFrame(columns=FEATURE_COLUMNS)

    if X_train.empty:
        print("ERROR: Training features (X_train) are empty. Cannot proceed.")
        return

    (
        X_train_processed,
        X_test_processed,
        y_train_processed,
        y_test_processed,
        X_test_processed_full,
    ) = preprocess_data(  # X_test_processed_full is processed X_test before y_test NaN cleaning
        X_train.copy(),
        X_test.copy()
        if not X_test.empty
        else pd.DataFrame(
            columns=X_train.columns
        ),  # Pass empty df with cols if X_test is empty
        y_train.copy(),
        y_test.copy() if not y_test.empty else pd.Series(dtype="float64"),
    )

    # For future predictions, we want to use data from ALL countries in YEAR_TEST,
    # not just those with non-NaN target. So, we re-process X_test_for_future_base_raw
    # (which is df_test_full_fe[FEATURE_COLUMNS])
    # For simplicity with current preprocess_data, we'll use X_test_processed_full
    # and align it with df_test_full_fe for country names for the 2025 prediction.
    # X_test_processed_full has undergone imputation and scaling using train-fit transformers.

    # This is the data that will serve as the "scenario" for 2025 predictions
    X_future_scenario_source_features = X_test_processed_full.copy()
    # Get corresponding country names for X_future_scenario_source_features
    # Assuming X_test_processed_full maintains index from df_test_full_fe[FEATURE_COLUMNS]
    df_future_scenario_source_context = df_test_full_fe[
        [COUNTRY_COLUMN]
    ].copy()  # Ensure index aligns
    if len(X_future_scenario_source_features) != len(df_future_scenario_source_context):
        print(
            f"Warning: Length mismatch between X_future_scenario_source_features ({len(X_future_scenario_source_features)}) and "
            f"df_future_scenario_source_context ({len(df_future_scenario_source_context)}). Country names might be incorrect for 2025 predictions."
        )
        # As a fallback, create a dummy country context if lengths differ significantly
        df_future_scenario_source_context = pd.DataFrame(
            {
                COUNTRY_COLUMN: [
                    f"Unknown_{i}"
                    for i in range(len(X_future_scenario_source_features))
                ]
            }
        )
    else:
        df_future_scenario_source_context = (
            df_future_scenario_source_context.reset_index(drop=True)
        )
        X_future_scenario_source_features = (
            X_future_scenario_source_features.reset_index(drop=True)
        )

    if X_train_processed.empty or y_train_processed.empty:
        print("Training data is empty after preprocessing. Cannot proceed.")
        return

    current_feature_names_processed = X_train_processed.columns.tolist()
    all_trained_models = {}
    all_results = {}
    sfs_selected_feature_names = None

    sfs_model, sfs_results, sfs_selected_feature_names = (
        perform_stepwise_selection_and_train(
            X_train_processed.copy(),
            y_train_processed.copy(),
            X_test_processed.copy() if not X_test_processed.empty else pd.DataFrame(),
            y_test_processed.copy()
            if not y_test_processed.empty
            else pd.Series(dtype="float64"),
            current_feature_names_processed,
            output_dir=OUTPUT_DIR,
        )
    )
    if sfs_model:
        all_trained_models["Linear Regression (SFS)"] = sfs_model
    if sfs_results:
        all_results["Linear Regression (SFS)"] = sfs_results

    trained_models_all_feat, results_all_feat = modeling_and_evaluation(
        X_train_processed,
        y_train_processed,
        X_test_processed if not X_test_processed.empty else pd.DataFrame(),
        y_test_processed if not y_test_processed.empty else pd.Series(dtype="float64"),
        current_feature_names_processed,
        output_dir=OUTPUT_DIR,
    )
    all_trained_models.update(trained_models_all_feat)
    all_results.update(results_all_feat)

    print("\n--- Combined Model Performance Summary (on YEAR_TEST) ---")
    # ... (summary print loop)

    # --- Analysis of Predictions on YEAR_TEST data ---
    if not X_test_processed.empty and not y_test_processed.empty and all_trained_models:
        # ... (select best model or default for analysis_of_predictions)
        # For example, analyze the SFS model if available
        model_to_analyze_name = (
            "Linear Regression (SFS)" if sfs_model else "Random Forest Regressor"
        )
        if model_to_analyze_name in all_trained_models:
            features_for_analysis = (
                sfs_selected_feature_names
                if model_to_analyze_name == "Linear Regression (SFS)"
                and sfs_selected_feature_names
                else current_feature_names_processed
            )
            analysis_of_predictions(
                all_trained_models[model_to_analyze_name],
                X_test_processed,
                y_test_processed,
                model_to_analyze_name,
                TARGET_COLUMN,
                YEAR_TEST,
                output_dir=OUTPUT_DIR,
                feature_names_for_model=features_for_analysis,
            )

    # --- Prediction for Future Year (e.g., 2025) ---
    print(f"\n--- Attempting Prediction for {YEAR_PREDICT_FUTURE} ---")
    model_for_future = None
    model_name_future = ""
    features_for_future_model = None

    if sfs_model and sfs_selected_feature_names:
        print(f"Using SFS model for {YEAR_PREDICT_FUTURE} prediction.")
        model_for_future = sfs_model
        model_name_future = "Linear Regression (SFS)"
        features_for_future_model = sfs_selected_feature_names
    elif "Random Forest Regressor" in all_trained_models:
        print(
            f"SFS model not available/suitable. Using Random Forest Regressor for {YEAR_PREDICT_FUTURE} prediction."
        )
        model_for_future = all_trained_models["Random Forest Regressor"]
        model_name_future = "Random Forest Regressor"
        features_for_future_model = (
            current_feature_names_processed  # RF uses all features it was trained on
        )
    else:
        print(
            f"No suitable model (SFS or RF) found for {YEAR_PREDICT_FUTURE} prediction."
        )

    if model_for_future and not X_future_scenario_source_features.empty:
        predict_for_future_year(
            model=model_for_future,
            X_future_scenario_features=X_future_scenario_source_features,  # Processed features from YEAR_TEST for all countries
            df_base_year_for_context=df_future_scenario_source_context,  # Country names from YEAR_TEST for all countries
            model_name=model_name_future,
            target_column_name=TARGET_COLUMN,
            future_year=YEAR_PREDICT_FUTURE,
            output_dir=FUTURE_PREDICTION_OUTPUT_DIR,
            country_column_name=COUNTRY_COLUMN,
            feature_names_for_model=features_for_future_model,
            base_year_actual_target=None,  # No actuals for future, could pass y_test from YEAR_TEST for context if X matches df_test_for_context
        )
    elif X_future_scenario_source_features.empty:
        print(
            f"Cannot make {YEAR_PREDICT_FUTURE} predictions as the scenario feature set (from {YEAR_TEST}) is empty."
        )
    else:
        print(f"Model for {YEAR_PREDICT_FUTURE} prediction is not available.")

    print("\n--- End of Analysis ---")


if __name__ == "__main__":
    # if not os.path.exists(FILE_PATH):
    #     print("error")
    #     raise Exeption ("shit")

    main()

    if plt.get_fignums():  # Check if any figures are open
        print("\nDisplaying plots. Close plot windows to end script.")
        plt.show(block=True)
    else:
        print("\nNo plots were generated or all plots were saved and closed.")

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
COUNTRY_COLUMN = "Country Name"  # Assuming this column might be useful for context, though not used in features

FILE_PATH = "assets/life_indicator_2008-2018.xlsx"
YEAR_TRAIN_START = 2008
YEAR_TRAIN_END = 2017
YEAR_TEST = 2018
OUTPUT_DIR = "results/task1_multiyear_before"


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
                    all_sheets_data[year_str] = None  # Store None if loading fails
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

    # if not any(all_sheets_data.values()):  # Check if any data was loaded
    #     print("No data sheets were successfully loaded.")
    #     return None
    return all_sheets_data


def engineer_features(df_train, df_test):
    """
    Creates new features from existing ones for both training and testing datasets.
    Updates the global FEATURE_COLUMNS list.
    """
    global FEATURE_COLUMNS
    new_features_created_local = []  # Use a local list to track features created in this call

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

    # Add unique new features to the global list
    # This ensures FEATURE_COLUMNS is updated only once with all unique new features from both sets
    for nf in new_features_created_local:
        if nf not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append(nf)

    if (
        new_features_created_local
    ):  # Check if any features were created in this specific call
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
    plt.figure(figsize=(16, 14))  # Adjusted size for potentially more features
    try:
        numeric_cols_for_corr = (
            df[feature_columns + [target_column]]
            .select_dtypes(include=np.number)
            .columns
        )
        correlation_matrix = df[numeric_cols_for_corr].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 8},
        )
        plt.title(f"Correlation Heatmap - {data_description}")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/correlation_heatmap_{data_description.replace(' ', '_').lower()}.png"
        )
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
        plt.savefig(
            f"{output_dir}/target_distribution_{data_description.replace(' ', '_').lower()}.png"
        )
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
    if X_test is not None and not X_test.empty and not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame if not empty")

    # Impute missing values in features using Median
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns

    imputer_X = SimpleImputer(strategy="median")

    X_train_imputed_numeric = imputer_X.fit_transform(X_train[numeric_cols_train])
    X_train_imputed_df = pd.DataFrame(
        X_train_imputed_numeric, columns=numeric_cols_train, index=X_train.index
    )
    for col in (
        X_train.columns
    ):  # Add back non-numeric columns if any (should not be for these features)
        if col not in numeric_cols_train:
            X_train_imputed_df[col] = X_train[col]
    X_train_imputed_df = X_train_imputed_df[X_train.columns]  # Reorder

    X_test_imputed_df = pd.DataFrame(columns=X_train.columns)  # Default empty
    if X_test is not None and not X_test.empty:
        numeric_cols_test = X_test.select_dtypes(include=np.number).columns
        # Ensure test set has same numeric columns as train for imputation
        cols_to_impute_test = [
            col for col in numeric_cols_train if col in numeric_cols_test
        ]

        if not cols_to_impute_test:
            print(
                "Warning: No common numeric columns found between X_train and X_test for imputation."
            )
            X_test_imputed_df = X_test.copy()  # No imputation possible on numeric
        else:
            X_test_imputed_numeric = imputer_X.transform(X_test[cols_to_impute_test])
            X_test_imputed_df_temp = pd.DataFrame(
                X_test_imputed_numeric, columns=cols_to_impute_test, index=X_test.index
            )
            # Merge imputed numeric with original non-numeric from X_test
            X_test_imputed_df = X_test.copy()
            for col in cols_to_impute_test:
                X_test_imputed_df[col] = X_test_imputed_df_temp[col]

        # Fill any columns present in train but not in test (e.g., after feature eng on train only) with median
        for col in X_train.columns:
            if col not in X_test_imputed_df.columns:
                # This scenario should ideally be handled by ensuring features are consistent
                print(
                    f"Warning: Column '{col}' from training data not in test data. Adding with NaNs then imputing."
                )
                X_test_imputed_df[col] = np.nan
        # Re-impute X_test with the full set of columns from X_train to handle newly added NaN columns
        # This assumes that any new columns are numeric or will be handled.
        # A more robust way is to ensure feature engineering consistency.
        if any(
            X_test_imputed_df[col].isnull().any()
            for col in numeric_cols_train
            if col in X_test_imputed_df.columns
        ):
            numeric_cols_test_reimpute = X_test_imputed_df.select_dtypes(
                include=np.number
            ).columns
            cols_to_reimpute_test = [
                col for col in numeric_cols_train if col in numeric_cols_test_reimpute
            ]
            if cols_to_reimpute_test:
                X_test_reimputed_numeric = imputer_X.transform(
                    X_test_imputed_df[cols_to_reimpute_test]
                )
                X_test_reimputed_df_final_temp = pd.DataFrame(
                    X_test_reimputed_numeric,
                    columns=cols_to_reimpute_test,
                    index=X_test_imputed_df.index,
                )
                for col in cols_to_reimpute_test:
                    X_test_imputed_df[col] = X_test_reimputed_df_final_temp[col]

    # Scale features
    scaler = StandardScaler()
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

    X_test_scaled_df = pd.DataFrame(columns=X_train.columns)  # Default empty
    if X_test is not None and not X_test.empty:
        # Ensure X_test_imputed_df has all columns that X_train_imputed_df has (for scaler)
        # and in the same order, particularly numeric_cols_train
        missing_cols_in_test_for_scaling = [
            col for col in numeric_cols_train if col not in X_test_imputed_df.columns
        ]
        if missing_cols_in_test_for_scaling:
            print(
                f"Warning: Test data is missing columns for scaling: {missing_cols_in_test_for_scaling}. Adding them with 0 or median."
            )
            for col in missing_cols_in_test_for_scaling:
                X_test_imputed_df[col] = (
                    0  # Or use another strategy like median of train column
                )
                # Example: X_test_imputed_df[col] = X_train_imputed_df[col].median()

        # Scale only the common numeric columns that were fit on
        cols_to_scale_test = [
            col for col in numeric_cols_train if col in X_test_imputed_df.columns
        ]
        X_test_scaled_numeric = scaler.transform(X_test_imputed_df[cols_to_scale_test])
        X_test_scaled_df_temp = pd.DataFrame(
            X_test_scaled_numeric,
            columns=cols_to_scale_test,
            index=X_test_imputed_df.index,
        )
        # Build X_test_scaled_df ensuring all original columns are present
        X_test_scaled_df = X_test_imputed_df.copy()
        for col in cols_to_scale_test:
            X_test_scaled_df[col] = X_test_scaled_df_temp[col]

        # Reorder X_test_scaled_df columns to match X_train_scaled_df
        X_test_scaled_df = X_test_scaled_df[X_train_scaled_df.columns]

    y_imputer = SimpleImputer(strategy="median")
    y_train_imputed = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_train_processed = pd.Series(
        y_train_imputed, index=y_train.index, name=y_train.name
    )

    y_test_cleaned = pd.Series(dtype="float64")
    X_test_scaled_df_cleaned = X_test_scaled_df.copy()

    if y_test is not None and not y_test.empty:
        y_test_nan_mask = ~pd.Series(y_test).isna()
        y_test_cleaned = y_test[y_test_nan_mask]
        X_test_scaled_df_cleaned = X_test_scaled_df[y_test_nan_mask]

        if not isinstance(y_test_cleaned, pd.Series):
            y_test_cleaned = pd.Series(
                y_test_cleaned,
                index=y_test[y_test_nan_mask].index
                if hasattr(y_test, "index")
                else None,
                name=y_test.name,
            )
    print("Preprocessing complete.")
    return (
        X_train_scaled_df,
        X_test_scaled_df_cleaned,
        y_train_processed,
        y_test_cleaned,
    )


def perform_stepwise_selection_and_train(
    X_train, y_train, X_test, y_test, feature_names, output_dir="."
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
    os.makedirs(output_dir, exist_ok=True)

    if X_train.empty or y_train.empty:
        print("Skipping SFS: Training data is empty.")
        return None, {}

    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train

    lr = LinearRegression()

    sfs = SFS(
        lr,
        k_features=(1, X_train_df.shape[1]),
        forward=True,
        floating=False,
        scoring="r2",
        cv=min(5, X_train_df.shape[0] if X_train_df.shape[0] > 1 else 2),  # Adjusted CV
        n_jobs=-1,
    )

    print("Fitting SFS model...")
    try:
        sfs = sfs.fit(X_train_df, y_train)
    except ValueError as ve:
        if "n_splits=" in str(
            ve
        ) and "cannot be greater than the number of samples" in str(ve):
            print(
                f"SFS Error: Not enough samples for CV folds. Samples: {X_train_df.shape[0]}. Trying with fewer folds."
            )
            try:
                sfs.cv = (
                    max(2, X_train_df.shape[0] - 1) if X_train_df.shape[0] > 1 else 0
                )  # Ensure cv > 1 or 0 for LOO
                if sfs.cv > 1:
                    sfs = sfs.fit(X_train_df, y_train)
                else:
                    print("Cannot perform SFS with CV less than 2. Skipping SFS.")
                    return None, {}
            except Exception as e_inner:
                print(f"Error during SFS fitting (retry): {e_inner}")
                return None, {}
        else:
            print(f"Error during SFS fitting: {ve}")
            return None, {}
    except Exception as e:
        print(f"Error during SFS fitting: {e}")
        return None, {}

    selected_features_indices = list(sfs.k_feature_idx_)
    if not selected_features_indices:  # Check if any features were selected
        print("SFS did not select any features. Aborting SFS model training.")
        return None, {}

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

        # Ensure only selected features are present in X_test_sfs and in correct order
        missing_sfs_cols_in_test = [
            col for col in selected_features_names if col not in X_test_df.columns
        ]
        if missing_sfs_cols_in_test:
            print(
                f"Warning: SFS selected features missing from test data: {missing_sfs_cols_in_test}. Cannot evaluate SFS model on test set."
            )
            return sfs_model, {
                "MSE": np.nan,
                "R²": np.nan,
                "Warning": "Missing SFS features in test data",
            }

        X_test_sfs = X_test_df[selected_features_names]
        predictions_sfs = sfs_model.predict(X_test_sfs)
        mse_sfs = mean_squared_error(y_test, predictions_sfs)
        r2_sfs = r2_score(y_test, predictions_sfs)
        results_sfs = {"MSE": mse_sfs, "R²": r2_sfs}
        print(
            f"Linear Regression (SFS) - Test MSE: {mse_sfs:.4f}, Test R²: {r2_sfs:.4f}"
        )

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
            plt.show()
        except Exception as e:
            print(f"Could not plot Linear Regression (SFS) coefficients: {e}")
    else:
        print(
            "Test data is empty or unavailable, skipping SFS model evaluation on test set."
        )
        results_sfs = {"MSE": np.nan, "R²": np.nan}

    return sfs_model, results_sfs


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
            "Skipping modeling evaluation on test set: Test data (X_test or y_test) is empty or unavailable. "
            "Models will be trained, but not evaluated on test data."
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
                # Ensure X_test has the same columns as X_train for prediction
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

                X_test_ordered = X_test[current_feature_names]  # Ensure column order
                predictions = model.predict(X_test_ordered)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results[name] = {"MSE": mse, "R²": r2}
                print(f"{name} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")
            else:
                results[name] = {"MSE": np.nan, "R²": np.nan}
                print(f"{name} - Test evaluation skipped.")

        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            if name in trained_models:
                del trained_models[name]
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
            plt.savefig(f"{output_dir}/lr_all_features_importances.png")
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
            plt.savefig(f"{output_dir}/rf_feature_importances.png")
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
            plt.savefig(f"{output_dir}/gb_feature_importances.png")
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
    output_dir=".",
    feature_names_for_model=None,  # For SFS model which might use a subset of features
):
    """Visualizes prediction differences and errors for a given model."""
    if model is None:
        print(f"Skipping prediction analysis for {model_name}: Model is None.")
        return
    if X_test is None or X_test.empty or y_test is None or y_test.empty:
        print(f"Skipping prediction analysis for {model_name}: Test data is empty.")
        return
    if not hasattr(model, "predict"):
        print(f"Skipping prediction analysis for {model_name}: Model cannot predict.")
        return

    print(f"\n--- c. Analysis of Predictions for {model_name} (Year {test_year}) ---")
    os.makedirs(output_dir, exist_ok=True)

    X_test_for_prediction = X_test
    if feature_names_for_model:  # If specific features are needed (e.g. for SFS model)
        missing_model_cols = [
            col for col in feature_names_for_model if col not in X_test.columns
        ]
        if missing_model_cols:
            print(
                f"Warning for {model_name} analysis: Test data missing model-specific columns: {missing_model_cols}. Cannot analyze."
            )
            return
        X_test_for_prediction = X_test[feature_names_for_model]

    try:
        predictions = model.predict(X_test_for_prediction)
    except Exception as e:
        print(f"Error during prediction with {model_name}: {e}")
        return

    y_test_np = (
        y_test.values.ravel()
        if isinstance(y_test, pd.Series)
        else np.array(y_test).ravel()
    )
    predictions_np = predictions.ravel()

    if len(y_test_np) != len(predictions_np):
        print(
            f"Length mismatch: y_test ({len(y_test_np)}), predictions ({len(predictions_np)}). Skipping."
        )
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, predictions_np, alpha=0.6, edgecolors="w", linewidth=0.5)
    min_val = (
        min(y_test_np.min(), predictions_np.min())
        if len(y_test_np) > 0 and len(predictions_np) > 0
        else 0
    )
    max_val = (
        max(y_test_np.max(), predictions_np.max())
        if len(y_test_np) > 0 and len(predictions_np) > 0
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
    plt.show()

    errors = y_test_np - predictions_np
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
    plt.show()

    errors_series = pd.Series(errors)
    print(f"\nSummary statistics for prediction errors ({model_name}):")
    print(errors_series.describe())

    mean_error = errors_series.mean()
    std_error = errors_series.std()

    if std_error == 0 or pd.isna(std_error):
        print("Standard deviation of errors is zero or NaN. Outlier detection skipped.")
        return

    outlier_threshold_upper = mean_error + 2 * std_error
    outlier_threshold_lower = mean_error - 2 * std_error

    outliers_mask = (errors_series > outlier_threshold_upper) | (
        errors_series < outlier_threshold_lower
    )
    if outliers_mask.any():
        print(
            f"\nPotential Outliers (errors > 2 std dev from mean error) for {model_name}:"
        )

        # Use y_test.index if y_test is a Series and its index corresponds to original df_test_full
        # This assumes X_test and y_test were aligned before this function
        outlier_indices_in_y_test = (
            y_test.index[outliers_mask]
            if isinstance(y_test, pd.Series)
            else np.where(outliers_mask)[0]
        )

        outlier_data = pd.DataFrame(
            {
                "Actual": y_test_np[outliers_mask],
                "Predicted": predictions_np[outliers_mask],
                "Error": errors[outliers_mask],
            },
            index=outlier_indices_in_y_test,
        )  # This index might be from y_test_cleaned
        print(outlier_data)
        print("\nPossible reasons for outliers:")
        print(
            f"- Unique national circumstances in {test_year} not captured by model trained on {YEAR_TRAIN_START}-{YEAR_TRAIN_END} data."
        )
        print("- Data quality issues for specific countries/features in the test year.")
        print("- Model limitations for these specific cases.")
    else:
        print(
            "\nNo significant outliers (errors > 2 std dev from mean error) detected."
        )


def main():
    global FEATURE_COLUMNS

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory for all results

    # --- Load Data ---
    all_loaded_data = load_all_relevant_data(
        FILE_PATH, YEAR_TRAIN_START, YEAR_TRAIN_END, YEAR_TEST
    )

    if all_loaded_data is None:
        print("\nExiting: Failed to load any data.")
        return

    # --- Aggregate Training Data ---
    df_train_list = []
    print(
        f"\n--- Aggregating Training Data from {YEAR_TRAIN_START} to {YEAR_TRAIN_END} ---"
    )
    for year in range(YEAR_TRAIN_START, YEAR_TRAIN_END + 1):
        year_str = str(year)
        if year_str in all_loaded_data and all_loaded_data[year_str] is not None:
            print(f"Adding data from year {year_str} to training set.")
            # Add a year column if it doesn't exist, can be useful for some analyses later (not used as feature here)
            if "Year" not in all_loaded_data[year_str].columns:
                all_loaded_data[year_str]["Year"] = year
            df_train_list.append(all_loaded_data[year_str])
        else:
            print(
                f"Warning: Data for training year {year_str} is missing or failed to load. Skipping."
            )

    if not df_train_list:
        print("\nExiting: No training data could be aggregated.")
        return
    df_train_full = pd.concat(df_train_list, ignore_index=True)
    print(f"Combined training data shape: {df_train_full.shape}")

    # --- Get Test Data ---
    df_test_full = all_loaded_data.get(str(YEAR_TEST))
    if df_test_full is None:
        print(
            f"Warning: Test data for year {YEAR_TEST} could not be loaded. Test set evaluation will be skipped."
        )
    else:
        if "Year" not in df_test_full.columns:  # Add Year column to test data too
            df_test_full["Year"] = YEAR_TEST
        print(f"Test data ({YEAR_TEST}) shape: {df_test_full.shape}")

    # --- Feature Engineering ---
    # Make copies to avoid SettingWithCopyWarning if df_train_full or df_test_full are slices
    df_train_full_fe = df_train_full.copy() if df_train_full is not None else None
    df_test_full_fe = df_test_full.copy() if df_test_full is not None else None

    # df_train_full_fe, df_test_full_fe = engineer_features(
    #     df_train_full_fe, df_test_full_fe
    # )

    # --- Data Understanding (on combined training data) ---
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

    # --- Prepare data for modeling ---
    if TARGET_COLUMN not in df_train_full_fe.columns or not all(
        col in df_train_full_fe.columns for col in FEATURE_COLUMNS
    ):
        print(
            f"Error: Target column '{TARGET_COLUMN}' or some feature columns (current list: {FEATURE_COLUMNS}) not found in combined training data."
        )
        return

    # Drop rows where target is NaN from training data *before* splitting
    df_train_full_fe.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df_train_full_fe.empty:
        print(
            f"Error: Combined training data is empty after dropping rows with NaN in '{TARGET_COLUMN}'."
        )
        return

    X_train = df_train_full_fe[FEATURE_COLUMNS]
    y_train = df_train_full_fe[TARGET_COLUMN]

    X_test = pd.DataFrame()
    y_test = pd.Series(dtype="float64")

    if df_test_full_fe is not None:
        missing_cols_test = [
            col for col in FEATURE_COLUMNS if col not in df_test_full_fe.columns
        ]
        if TARGET_COLUMN not in df_test_full_fe.columns or missing_cols_test:
            print(
                f"Warning: Target column '{TARGET_COLUMN}' or some feature columns "
                f"(missing: {missing_cols_test if missing_cols_test else 'None'}) "
                f"not found in testing data for year {YEAR_TEST} after feature engineering."
            )
            # We will proceed but X_test, y_test will be empty, skipping test evaluation
        else:
            X_test = df_test_full_fe[FEATURE_COLUMNS]
            y_test = df_test_full_fe[TARGET_COLUMN]
    else:
        print(
            f"Test data for {YEAR_TEST} was not loaded or is empty. Evaluation on test set will be skipped."
        )

    if X_train.empty:
        print("ERROR: Training features (X_train) are empty. Cannot proceed.")
        return

    # --- Preprocessing ---
    X_train_processed, X_test_processed, y_train_processed, y_test_processed = (
        preprocess_data(
            X_train.copy(),  # Pass copies to avoid issues
            X_test.copy() if not X_test.empty else None,
            y_train.copy(),
            y_test.copy() if not y_test.empty else None,
        )
    )

    if X_train_processed.empty or y_train_processed.empty:
        print("Training data is empty after preprocessing. Cannot proceed.")
        return

    current_feature_names_processed = X_train_processed.columns.tolist()

    all_trained_models = {}
    all_results = {}

    # --- Stepwise Forward Selection (SFS) ---
    sfs_model, sfs_results = perform_stepwise_selection_and_train(
        X_train_processed.copy(),
        y_train_processed.copy(),
        X_test_processed.copy()
        if X_test_processed is not None and not X_test_processed.empty
        else pd.DataFrame(),
        y_test_processed.copy()
        if y_test_processed is not None and not y_test_processed.empty
        else pd.Series(dtype="float64"),
        current_feature_names_processed,
        output_dir=OUTPUT_DIR,
    )
    if sfs_model:
        all_trained_models["Linear Regression (SFS)"] = sfs_model
    if sfs_results:
        all_results["Linear Regression (SFS)"] = sfs_results

    # --- Modeling and Evaluation (on all features) ---
    trained_models_all_feat, results_all_feat = modeling_and_evaluation(
        X_train_processed,
        y_train_processed,
        X_test_processed
        if X_test_processed is not None and not X_test_processed.empty
        else pd.DataFrame(),
        y_test_processed
        if y_test_processed is not None and not y_test_processed.empty
        else pd.Series(dtype="float64"),
        current_feature_names_processed,
        output_dir=OUTPUT_DIR,
    )
    all_trained_models.update(trained_models_all_feat)
    all_results.update(results_all_feat)

    print("\n--- Combined Model Performance Summary ---")
    if all_results:
        for name, metrics in all_results.items():
            if "Error" in metrics and metrics["Error"]:
                print(f"{name} - Error: {metrics['Error']}")
            elif pd.isna(metrics.get("MSE", np.nan)) and pd.isna(
                metrics.get("R²", np.nan)
            ):
                print(
                    f"{name} - Trained, but no test evaluation performed or evaluation failed."
                )
            else:
                print(
                    f"{name} - Test MSE: {metrics.get('MSE', np.nan):.4f}, Test R²: {metrics.get('R²', np.nan):.4f}"
                )
    else:
        print("No models were successfully evaluated.")

    # --- Analysis of Predictions ---
    # Perform analysis only if X_test_processed and y_test_processed are valid
    if (
        X_test_processed is not None
        and not X_test_processed.empty
        and y_test_processed is not None
        and not y_test_processed.empty
        and all_trained_models
    ):
        best_model_name = None
        best_r2 = -float("inf")

        if all_results:
            for name, metrics in all_results.items():
                if pd.notna(metrics.get("R²")) and metrics["R²"] > best_r2:
                    best_r2 = metrics["R²"]
                    best_model_name = name

        model_to_analyze_name = None
        sfs_selected_feature_names = None

        if best_model_name and best_model_name in all_trained_models:
            model_to_analyze_name = best_model_name
            print(
                f"\nAnalyzing predictions for the best model based on R² on Test Set: {model_to_analyze_name} (R²: {best_r2:.4f})"
            )
        elif "Random Forest Regressor" in all_trained_models:
            model_to_analyze_name = "Random Forest Regressor"
            print(
                "\nCould not determine best model by R² or no test results. Defaulting analysis to Random Forest Regressor."
            )
        elif all_trained_models:
            model_to_analyze_name = list(all_trained_models.keys())[0]
            print(
                f"\nCould not determine best model by R². Analyzing first available model: {model_to_analyze_name}"
            )

        if model_to_analyze_name and model_to_analyze_name in all_trained_models:
            model_for_analysis = all_trained_models[model_to_analyze_name]
            features_for_this_model = (
                current_feature_names_processed  # Default to all processed features
            )

            if (
                model_to_analyze_name == "Linear Regression (SFS)"
                and sfs_model is not None
            ):
                # sfs_model.feature_names_in_ might not be directly available if sfs object was used for k_feature_names_
                # We need the names used for training the final SFS model
                if hasattr(
                    sfs_model, "feature_names_in_"
                ):  # Scikit-learn models usually have this
                    sfs_selected_feature_names = list(sfs_model.feature_names_in_)
                    features_for_this_model = sfs_selected_feature_names
                elif (
                    "Linear Regression (SFS)" in all_results
                    and "features" in all_results["Linear Regression (SFS)"]
                ):
                    # Fallback: if we stored them during SFS (not currently done, but good idea)
                    sfs_selected_feature_names = all_results["Linear Regression (SFS)"][
                        "features"
                    ]
                    features_for_this_model = sfs_selected_feature_names
                else:  # Re-fetch from sfs object if still in scope and fitted
                    try:
                        # sfs object from perform_stepwise_selection_and_train should have k_feature_names_
                        sfs_selected_feature_names = list(
                            sfs.k_feature_names_
                        )  # sfs object from SFS function
                        features_for_this_model = sfs_selected_feature_names
                        print(
                            f"Using SFS selected features for analysis: {features_for_this_model}"
                        )
                    except NameError:  # sfs object might not be in this scope directly
                        print(
                            "Warning: Could not retrieve SFS selected features for detailed analysis. Using all features, results might be off for SFS model."
                        )
                    except AttributeError:
                        print(
                            "Warning: SFS object does not have k_feature_names_. Analysis might be off for SFS model."
                        )

            analysis_of_predictions(
                model_for_analysis,
                X_test_processed,  # X_test_processed should have all columns. The function will subset if needed.
                y_test_processed,
                model_to_analyze_name,
                TARGET_COLUMN,
                YEAR_TEST,
                output_dir=OUTPUT_DIR,
                feature_names_for_model=features_for_this_model,  # Pass the correct feature set
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
    try:
        # Try to read sheet names to check if file is a valid Excel file
        pd.ExcelFile(FILE_PATH).sheet_names
        print(f"Found existing file: '{FILE_PATH}'. Will use it.")        
    except Exception as e:
        print(f"An error occurred while checking or creating the dummy file: {e}")
        print(
            "Proceeding, but the script might fail if the file is not correctly formatted or accessible."
        )

    main()
    if plt.get_fignums():
        print("\nDisplaying plots. Close plot windows to end script.")
        plt.show(block=True)  # block=True will wait for plots to be closed
    else:
        print("\nNo plots were generated or all plots were saved and closed.")

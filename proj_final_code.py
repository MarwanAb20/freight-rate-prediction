"""
Freight Rate Prediction Challenge
Final ML Pipeline

Project:
    Freight Rate Prediction

Pipeline:
1. Load data
2. Data quality checks
3. Chronological validation
4. Compare raw/log target models
5. Select model by validation MAE
6. Train final model on all labeled data
7. Predict validation.csv
8. Predict December chart inputs
9. Validate output files

Important:
CatBoost categorical features are explicitly specified.
"""

import os
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train-test.csv"
)

VALIDATION_PATH = os.path.join(
    DATA_DIR,
    "validation.csv"
)

TEMPLATE_PATH = os.path.join(
    DATA_DIR,
    "validation-predictions-template.csv"
)

DECEMBER_PATH = os.path.join(
    DATA_DIR,
    "december-chart-inputs.csv"
)

VALIDATION_OUTPUT = os.path.join(
    BASE_DIR,
    "validation_predictions.csv"
)

RANDOM_SEED = 42


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(
    df,
    use_date_features=True
):
    """
    Create model features.

    Categorical:
        pickup
        delivery
        equipment
        route

    Numerical:
        coordinates
        distance
        weight
        market_index
        quote_signal
        engineered numeric features
    """

    x = df.copy()

    # --------------------------------------------------------
    # 1. Remove target / ID / previous prediction
    # --------------------------------------------------------

    columns_to_drop = [
        "posted_rate",
        "predicted_rate",
        "load_id"
    ]

    for col in columns_to_drop:

        if col in x.columns:

            x = x.drop(
                columns=col
            )

    # --------------------------------------------------------
    # 2. Optional numerical columns
    # --------------------------------------------------------

    optional_numeric = [
        "market_index",
        "quote_signal",
        "weight"
    ]

    for col in optional_numeric:

        if col not in x.columns:

            x[col] = np.nan

    # --------------------------------------------------------
    # 3. Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal"
    ]

    for col in numeric_columns:

        if col in x.columns:

            x[col] = pd.to_numeric(
                x[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # 4. Weight cleaning
    # --------------------------------------------------------

    x["weight"] = x["weight"].abs()

    # --------------------------------------------------------
    # 5. Geographic features
    # --------------------------------------------------------

    geo_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    for col in geo_columns:

        if col not in x.columns:

            x[col] = np.nan

    x["lat_diff"] = (
        x["delivery_lat"]
        - x["pickup_lat"]
    )

    x["lon_diff"] = (
        x["delivery_lon"]
        - x["pickup_lon"]
    )

    # --------------------------------------------------------
    # 6. Distance transformation
    # --------------------------------------------------------

    x["distance_log"] = np.log1p(
        x["distance"].clip(
            lower=0
        )
    )

    # --------------------------------------------------------
    # 7. Missing indicators
    # --------------------------------------------------------

    x["weight_missing"] = (
        x["weight"]
        .isna()
        .astype(int)
    )

    x["market_index_missing"] = (
        x["market_index"]
        .isna()
        .astype(int)
    )

    x["quote_signal_missing"] = (
        x["quote_signal"]
        .isna()
        .astype(int)
    )

    # --------------------------------------------------------
    # 8. Route
    # --------------------------------------------------------

    if (
        "pickup" in x.columns
        and "delivery" in x.columns
    ):

        x["route"] = (
            x["pickup"].astype(str)
            + " -> "
            + x["delivery"].astype(str)
        )

    # --------------------------------------------------------
    # 9. Date features
    # --------------------------------------------------------

    if "date" in x.columns:

        x["date"] = pd.to_datetime(
            x["date"],
            errors="coerce"
        )

        if use_date_features:

            x["month"] = (
                x["date"].dt.month
            )

            x["dayofweek"] = (
                x["date"].dt.dayofweek
            )

            x["dayofmonth"] = (
                x["date"].dt.day
            )

            x["dayofyear"] = (
                x["date"].dt.dayofyear
            )

            x["is_weekend"] = (
                x["date"].dt.dayofweek >= 5
            ).astype(int)

            # Cyclical month

            x["month_sin"] = np.sin(
                2 * np.pi * x["month"] / 12
            )

            x["month_cos"] = np.cos(
                2 * np.pi * x["month"] / 12
            )

            # Cyclical day of week

            x["dow_sin"] = np.sin(
                2 * np.pi * x["dayofweek"] / 7
            )

            x["dow_cos"] = np.cos(
                2 * np.pi * x["dayofweek"] / 7
            )

        x = x.drop(
            columns=["date"]
        )

    # --------------------------------------------------------
    # 10. Handle categorical columns
    # --------------------------------------------------------

    for col in CATEGORICAL_COLUMNS:

        if col in x.columns:

            # CatBoost requires categorical
            # values to be strings / non-null.

            x[col] = (
                x[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    # --------------------------------------------------------
    # 11. Convert booleans
    # --------------------------------------------------------

    bool_columns = (
        x.select_dtypes(
            include=["bool"]
        ).columns
    )

    for col in bool_columns:

        x[col] = x[col].astype(int)

    return x


# ============================================================
# GET CATEGORICAL FEATURE NAMES
# ============================================================

def get_categorical_features(
    feature_columns
):
    """
    Return categorical feature names
    that actually exist in the dataset.
    """

    return [
        col
        for col in CATEGORICAL_COLUMNS
        if col in feature_columns
    ]


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    iterations
):

    return CatBoostRegressor(

        iterations=iterations,

        learning_rate=0.04,

        depth=6,

        loss_function="RMSE",

        eval_metric="MAE",

        l2_leaf_reg=5,

        random_seed=RANDOM_SEED,

        verbose=False,

        allow_writing_files=False
    )


# ============================================================
# CHRONOLOGICAL EXPERIMENT
# ============================================================

def run_experiment(
    train_df,
    model_name,
    use_log_target,
    use_date_features
):

    print("\n" + "-" * 65)

    print(
        f"MODEL: {model_name}"
    )

    print("-" * 65)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_date = pd.Timestamp(
        "2025-10-01"
    )

    train_part = train_df[
        train_df["date"] < split_date
    ].copy()

    valid_part = train_df[
        train_df["date"] >= split_date
    ].copy()

    print(
        f"Training period : "
        f"{train_part['date'].min().date()} -> "
        f"{train_part['date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{valid_part['date'].min().date()} -> "
        f"{valid_part['date'].max().date()}"
    )

    print(
        f"Training rows   : "
        f"{len(train_part):,}"
    )

    print(
        f"Validation rows : "
        f"{len(valid_part):,}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = make_features(
        train_part,
        use_date_features
    )

    X_valid = make_features(
        valid_part,
        use_date_features
    )

    # EXACT same columns
    X_valid = X_valid.reindex(
        columns=X_train.columns
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_features = (
        get_categorical_features(
            X_train.columns
        )
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_train = (
        train_part["posted_rate"]
        .values
    )

    y_valid = (
        valid_part["posted_rate"]
        .values
    )

    if use_log_target:

        y_train_model = np.log1p(
            y_train
        )

        y_valid_model = np.log1p(
            y_valid
        )

    else:

        y_train_model = y_train
        y_valid_model = y_valid

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = create_model(
        iterations=1500
    )

    model.fit(

        X_train,

        y_train_model,

        cat_features=categorical_features,

        eval_set=(
            X_valid,
            y_valid_model
        ),

        use_best_model=True,

        verbose=False
    )

    # --------------------------------------------------------
    # Best iteration
    # --------------------------------------------------------

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):

        best_iteration = 1499

    best_iterations = (
        best_iteration + 1
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_model = model.predict(
        X_valid
    )

    if use_log_target:

        predictions = np.expm1(
            predictions_model
        )

    else:

        predictions = predictions_model

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    print(
        f"Features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Categorical features: "
        f"{categorical_features}"
    )

    print(
        "Target transformation: "
        + (
            "log1p"
            if use_log_target
            else "raw"
        )
    )

    print(
        "Date features: "
        + (
            "yes"
            if use_date_features
            else "no"
        )
    )

    print(
        f"MAE           : "
        f"${mae:,.2f}"
    )

    print(
        f"RMSE          : "
        f"${rmse:,.2f}"
    )

    print(
        f"R²            : "
        f"{r2:.4f}"
    )

    print(
        f"Best iteration: "
        f"{best_iterations}"
    )

    return {

        "name": model_name,

        "use_log_target":
            use_log_target,

        "use_date_features":
            use_date_features,

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "best_iterations":
            best_iterations,

        "feature_columns":
            list(X_train.columns),

        "categorical_features":
            categorical_features
    }


# ============================================================
# MODEL SELECTION
# ============================================================

def select_model(
    train_df
):

    print("\n" + "=" * 65)
    print("MODEL SELECTION")
    print("=" * 65)

    experiments = [

        {
            "name":
                "Raw target + core/date features",

            "use_log_target":
                False,

            "use_date_features":
                True
        },

        {
            "name":
                "Log target + core/date features",

            "use_log_target":
                True,

            "use_date_features":
                True
        },

        {
            "name":
                "Log target + no date features",

            "use_log_target":
                True,

            "use_date_features":
                False
        }
    ]

    results = []

    for experiment in experiments:

        result = run_experiment(

            train_df=train_df,

            model_name=experiment[
                "name"
            ],

            use_log_target=experiment[
                "use_log_target"
            ],

            use_date_features=experiment[
                "use_date_features"
            ]
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    display_df = results_df[
        [
            "name",
            "mae",
            "rmse",
            "r2",
            "best_iterations"
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "MAE",
        "RMSE",
        "R²",
        "Best Iteration"
    ]

    print(
        display_df
        .sort_values("MAE")
        .to_string(
            index=False,
            formatters={
                "MAE":
                    lambda x:
                    f"${x:,.2f}",

                "RMSE":
                    lambda x:
                    f"${x:,.2f}",

                "R²":
                    lambda x:
                    f"{x:.4f}"
            }
        )
    )

    # --------------------------------------------------------
    # Select lowest MAE
    # --------------------------------------------------------

    selected = min(
        results,
        key=lambda x: x["mae"]
    )

    print("\n" + "=" * 65)
    print("SELECTED MODEL")
    print("=" * 65)

    print(
        f"Model           : "
        f"{selected['name']}"
    )

    print(
        f"Validation MAE  : "
        f"${selected['mae']:,.2f}"
    )

    print(
        f"Validation RMSE : "
        f"${selected['rmse']:,.2f}"
    )

    print(
        f"Validation R²   : "
        f"{selected['r2']:.4f}"
    )

    print(
        f"Best iteration  : "
        f"{selected['best_iterations']}"
    )

    return selected


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model(
    train_df,
    selected_config
):

    print("\n" + "=" * 65)
    print("FINAL MODEL TRAINING")
    print("=" * 65)

    use_log_target = (
        selected_config[
            "use_log_target"
        ]
    )

    use_date_features = (
        selected_config[
            "use_date_features"
        ]
    )

    iterations = (
        selected_config[
            "best_iterations"
        ]
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = make_features(

        train_df,

        use_date_features
    )

    feature_columns = list(
        X_train.columns
    )

    categorical_features = (
        get_categorical_features(
            feature_columns
        )
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = (
        train_df["posted_rate"]
        .values
    )

    if use_log_target:

        y_model = np.log1p(y)

    else:

        y_model = y

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        iterations=iterations
    )

    model.fit(

        X_train,

        y_model,

        cat_features=categorical_features,

        verbose=False
    )

    print(
        f"Training rows : "
        f"{len(X_train):,}"
    )

    print(
        f"Features      : "
        f"{len(feature_columns)}"
    )

    print(
        f"Categorical   : "
        f"{categorical_features}"
    )

    print(
        "Target        : "
        + (
            "log1p(posted_rate)"
            if use_log_target
            else "posted_rate"
        )
    )

    print(
        f"Iterations    : "
        f"{iterations}"
    )

    print(
        "Final model trained successfully."
    )

    return (
        model,
        feature_columns
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_rates(

    model,

    df,

    feature_columns,

    use_log_target,

    use_date_features

):

    X = make_features(

        df,

        use_date_features
    )

    # EXACT SAME FEATURES
    X = X.reindex(

        columns=feature_columns,

        fill_value=np.nan
    )

    categorical_features = (
        get_categorical_features(
            feature_columns
        )
    )

    predictions_model = (
        model.predict(X)
    )

    if use_log_target:

        predictions = np.expm1(
            predictions_model
        )

    else:

        predictions = predictions_model

    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


# ============================================================
# VALIDATION OUTPUT
# ============================================================

def create_validation_predictions(

    model,

    feature_columns,

    selected_config,

    validation_df

):

    print("\n" + "=" * 65)
    print("FINAL VALIDATION PREDICTIONS")
    print("=" * 65)

    predictions = predict_rates(

        model=model,

        df=validation_df,

        feature_columns=feature_columns,

        use_log_target=
            selected_config[
                "use_log_target"
            ],

        use_date_features=
            selected_config[
                "use_date_features"
            ]
    )

    # --------------------------------------------------------
    # Official template
    # --------------------------------------------------------

    template = pd.read_csv(
        TEMPLATE_PATH
    )

    if len(template) != 12000:

        raise ValueError(
            "Template must contain "
            "exactly 12000 rows."
        )

    if len(predictions) != len(template):

        raise ValueError(
            "Prediction count does not "
            "match template."
        )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    if not np.array_equal(

        template["load_id"].values,

        validation_df["load_id"].values

    ):

        raise ValueError(
            "Validation IDs do not match "
            "template IDs."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = template[
        ["load_id"]
    ].copy()

    output[
        "predicted_rate"
    ] = predictions

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if output["load_id"].duplicated().any():

        raise ValueError(
            "Duplicate load_id detected."
        )

    if output[
        "predicted_rate"
    ].isna().any():

        raise ValueError(
            "NaN predictions detected."
        )

    if (
        output[
            "predicted_rate"
        ] <= 0
    ).any():

        raise ValueError(
            "Non-positive predictions detected."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(

        VALIDATION_OUTPUT,

        index=False
    )

    print(
        f"Predictions generated: "
        f"{len(output):,}"
    )

    print(
        f"Prediction range: "
        f"${output['predicted_rate'].min():,.2f}"
        f" - "
        f"${output['predicted_rate'].max():,.2f}"
    )

    print(
        f"Saved: "
        f"{VALIDATION_OUTPUT}"
    )


# ============================================================
# DECEMBER COORDINATES
# ============================================================

def add_december_coordinates(

    december_df,

    train_df

):

    december = (
        december_df.copy()
    )

    # --------------------------------------------------------
    # Pickup lookup
    # --------------------------------------------------------

    pickup_lookup = (

        train_df[

            [
                "pickup",
                "pickup_lat",
                "pickup_lon"
            ]

        ]

        .dropna()

        .drop_duplicates(
            subset=["pickup"]
        )

        .set_index("pickup")
    )

    # --------------------------------------------------------
    # Delivery lookup
    # --------------------------------------------------------

    delivery_lookup = (

        train_df[

            [
                "delivery",
                "delivery_lat",
                "delivery_lon"
            ]

        ]

        .dropna()

        .drop_duplicates(
            subset=["delivery"]
        )

        .set_index("delivery")
    )

    # --------------------------------------------------------
    # Map coordinates
    # --------------------------------------------------------

    december["pickup_lat"] = (
        december["pickup"]
        .map(
            pickup_lookup[
                "pickup_lat"
            ]
        )
    )

    december["pickup_lon"] = (
        december["pickup"]
        .map(
            pickup_lookup[
                "pickup_lon"
            ]
        )
    )

    december["delivery_lat"] = (
        december["delivery"]
        .map(
            delivery_lookup[
                "delivery_lat"
            ]
        )
    )

    december["delivery_lon"] = (
        december["delivery"]
        .map(
            delivery_lookup[
                "delivery_lon"
            ]
        )
    )

    return december


# ============================================================
# DECEMBER OUTPUT
# ============================================================

def create_december_predictions(

    model,

    feature_columns,

    selected_config,

    december_df,

    train_df

):

    print("\n" + "=" * 65)
    print("DECEMBER PREDICTIONS")
    print("=" * 65)

    # --------------------------------------------------------
    # Remove previous predictions
    # --------------------------------------------------------

    if "predicted_rate" in december_df.columns:

        december_df = (
            december_df.drop(
                columns=["predicted_rate"]
            )
        )

    # --------------------------------------------------------
    # Add coordinates
    # --------------------------------------------------------

    december = (
        add_december_coordinates(
            december_df,
            train_df
        )
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = predict_rates(

        model=model,

        df=december,

        feature_columns=feature_columns,

        use_log_target=
            selected_config[
                "use_log_target"
            ],

        use_date_features=
            selected_config[
                "use_date_features"
            ]
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = december.copy()

    output[
        "predicted_rate"
    ] = predictions

    required_columns = [

        "pickup",

        "delivery",

        "distance",

        "equipment",

        "weight",

        "date",

        "predicted_rate"
    ]

    output = output[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if len(output) != 31:

        raise ValueError(
            "December output must contain "
            "31 rows."
        )

    output["date"] = pd.to_datetime(
        output["date"],
        errors="coerce"
    )

    unique_dates = (
        output["date"]
        .dt.date
        .nunique()
    )

    if unique_dates != 31:

        raise ValueError(
            "December must contain "
            "31 unique dates."
        )

    if output[
        "predicted_rate"
    ].isna().any():

        raise ValueError(
            "NaN December predictions."
        )

    if (
        output[
            "predicted_rate"
        ] <= 0
    ).any():

        raise ValueError(
            "Non-positive December "
            "predictions."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(

        DECEMBER_PATH,

        index=False
    )

    print(
        f"December rows: "
        f"{len(output)}"
    )

    print(
        f"Unique dates: "
        f"{unique_dates}"
    )

    print(
        f"Prediction range: "
        f"${output['predicted_rate'].min():,.2f}"
        f" - "
        f"${output['predicted_rate'].max():,.2f}"
    )

    print(
        f"Saved: "
        f"{DECEMBER_PATH}"
    )

    print("\nDecember predictions:")

    print(
        output[
            [
                "date",
                "predicted_rate"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "FREIGHT RATE PREDICTION PIPELINE"
    )

    print("=" * 65)

    # ========================================================
    # LOAD
    # ========================================================

    print("\n[1] Loading data...")

    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    train["date"] = pd.to_datetime(
        train["date"],
        errors="coerce"
    )

    validation["date"] = pd.to_datetime(
        validation["date"],
        errors="coerce"
    )

    print(
        f"Train      : "
        f"{train.shape}"
    )

    print(
        f"Validation : "
        f"{validation.shape}"
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n[2] Data quality...")

    missing = train.isna().sum()

    missing = missing[
        missing > 0
    ]

    print(
        "\nMissing values in train:"
    )

    if len(missing) > 0:

        print(missing)

    else:

        print("No missing values.")

    print(
        f"\nDuplicate train rows: "
        f"{train.duplicated().sum()}"
    )

    print(
        f"Duplicate train IDs: "
        f"{train['load_id'].duplicated().sum()}"
    )

    print(
        f"Duplicate validation IDs: "
        f"{validation['load_id'].duplicated().sum()}"
    )

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    print(
        "\n[3] Model selection..."
    )

    selected_config = (
        select_model(train)
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print(
        "\n[4] Final model training..."
    )

    final_model, feature_columns = (
        train_final_model(
            train,
            selected_config
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n[5] Generating validation predictions..."
    )

    create_validation_predictions(

        final_model,

        feature_columns,

        selected_config,

        validation
    )

    # ========================================================
    # DECEMBER
    # ========================================================

    print(
        "\n[6] Generating December predictions..."
    )

    december = pd.read_csv(
        DECEMBER_PATH
    )

    create_december_predictions(

        final_model,

        feature_columns,

        selected_config,

        december,

        train
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n" + "=" * 65)

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print("=" * 65)

    print("\nGenerated:")

    print(
        "validation_predictions.csv"
    )

    print(
        "data/december-chart-inputs.csv"
    )

    print("\nRun scorer:")

    print(
        "python score.py "
        "--predictions validation_predictions.csv "
        "--december-predictions "
        "data/december-chart-inputs.csv"
    )

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
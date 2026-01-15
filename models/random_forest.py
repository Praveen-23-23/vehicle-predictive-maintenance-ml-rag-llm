import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/rf_model.joblib"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Target"


def train_and_save_model(csv_path):
    """
    Train the Random Forest model ONCE and save it to disk.
    """
    df = pd.read_csv(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("✅ Random Forest model trained and saved")



def load_and_predict(csv_path):
    """
    Load the trained model and run inference ONLY.
    """
    df = pd.read_csv(csv_path)

    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]

    result_df = X.copy()
    result_df["Failure_Probability"] = model.predict_proba(X)[:, 1]

    return result_df

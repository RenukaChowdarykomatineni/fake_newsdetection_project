"""
Fake News Detection

This script trains a text classification model to detect
fake vs real news articles using a Kaggle-style dataset
with two files:

    data/Fake.csv  -> fake news articles
    data/True.csv  -> real news articles

Label convention:
    0 = FAKE
    1 = REAL

Pipeline:
1. Load and combine Fake.csv + True.csv, add a 'label' column.
2. Prepare text and labels (combine title + text).
3. Build a model: TfidfVectorizer -> LogisticRegression.
4. Evaluate: accuracy, confusion matrix, precision, recall, F1.
5. Show some false positives & false negatives.
6. (Optional) Let the user type news text and get a prediction.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

RANDOM_STATE = 42

# The two Kaggle files you placed in data/
FAKE_PATH = "data/Fake.csv"
REAL_PATH = "data/True.csv"


# ----------------------------------------------------------------------
# PRETTY PRINT HELPERS
# ----------------------------------------------------------------------


def print_line(char: str = "-", length: int = 70) -> None:
    """Print a horizontal line."""
    print(char * length)


def print_title(title: str) -> None:
    """Print a big section title."""
    print_line("=")
    print(title)
    print_line("=")


def print_subtitle(subtitle: str) -> None:
    """Print a smaller section subtitle."""
    print(f"\n>> {subtitle}")
    print_line("-")


# ----------------------------------------------------------------------
# STEP 1: LOAD + COMBINE DATA
# ----------------------------------------------------------------------


def load_data(fake_path: str, real_path: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, add a label column, and combine.

    label = 0 -> FAKE (from Fake.csv)
    label = 1 -> REAL (from True.csv)
    """
    print_title("STEP 1: LOAD DATA")

    # Load fake news
    print_subtitle("Loading FAKE news")
    print(f"Reading: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0  # 0 = FAKE
    print(fake_df.head())

    # Load real news
    print_subtitle("Loading REAL news")
    print(f"Reading: {real_path}")
    real_df = pd.read_csv(real_path)
    real_df["label"] = 1  # 1 = REAL
    print(real_df.head())

    # Combine into one DataFrame
    df = pd.concat([fake_df, real_df], ignore_index=True)

    print_subtitle("Combined dataset info")
    print(df.info())

    print_subtitle("Label distribution (0 = FAKE, 1 = REAL)")
    print(df["label"].value_counts())

    return df


# ----------------------------------------------------------------------
# STEP 2: PREPARE TEXT AND LABELS
# ----------------------------------------------------------------------


def detect_text_column(df: pd.DataFrame) -> str:
    """
    Find or create a text column to feed into the model.

    For this dataset we usually have:
        'title' and 'text'
    We will combine them into one 'combined_text' column.
    """
    has_title = any(c in df.columns for c in ["title", "Title"])
    has_text = any(c in df.columns for c in ["text", "Text"])

    if has_title and has_text:
        print_subtitle("Combining 'title' and 'text' into 'combined_text'")
        title_col = "title" if "title" in df.columns else "Title"
        text_col = "text" if "text" in df.columns else "Text"

        df["combined_text"] = (
            df[title_col].fillna("").astype(str)
            + " "
            + df[text_col].fillna("").astype(str)
        )
        return "combined_text"

    # fallback if only one text column
    for col in ["text", "Text", "content", "article", "news"]:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a text column. Expected 'title' + 'text', "
        "or one of: text, Text, content, article, news."
    )


def prepare_features_and_labels(df: pd.DataFrame):
    """
    Prepare X (text) and y (labels) for training.

    Requires:
        - 'label' column with 0 = FAKE, 1 = REAL
        - some text column (we detect/construct it)
    """
    print_title("STEP 2: PREPARE TEXT AND LABELS")

    if "label" not in df.columns:
        raise ValueError(
            "Expected a 'label' column. Did you call load_data() first?"
        )

    text_col = detect_text_column(df)
    label_col = "label"

    print_subtitle("Using columns")
    print(f"Text column : {text_col}")
    print(f"Label column: {label_col}")
    print("Label meaning: 0 = FAKE, 1 = REAL")

    # Simple text cleaning
    df[text_col] = df[text_col].fillna("").astype(str).str.lower()

    X = df[text_col]
    y = df[label_col]

    print_subtitle("Final shapes")
    print(f"X (text) shape: {X.shape}")
    print(f"y (labels) shape: {y.shape}")

    return X, y


# ----------------------------------------------------------------------
# STEP 3: BUILD MODEL PIPELINE
# ----------------------------------------------------------------------


def build_model_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline:

        Raw text
          -> TfidfVectorizer (embeddings)
          -> LogisticRegression (classifier)
    """
    print_title("STEP 3: BUILD MODEL PIPELINE")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Pipeline created: TfidfVectorizer -> LogisticRegression")
    return pipeline


# ----------------------------------------------------------------------
# STEP 4: EVALUATION + MISCLASSIFICATION ANALYSIS
# ----------------------------------------------------------------------


def print_confusion_matrix(cm: np.ndarray) -> None:
    """
    Pretty-print confusion matrix for binary classification.

    We use label convention:
        0 = FAKE
        1 = REAL

    confusion_matrix(y_true, y_pred) returns:

                  Pred 0      Pred 1
      True 0  ->    tn          fp
      True 1  ->    fn          tp
    """
    tn, fp, fn, tp = cm.ravel()

    print_subtitle("Confusion Matrix (rows = Actual, cols = Predicted)")
    print("                 Predicted FAKE   Predicted REAL")
    print(f"Actual FAKE (0)     {tn:7d}          {fp:7d}")
    print(f"Actual REAL (1)     {fn:7d}          {tp:7d}")
    print("\nBreakdown:")
    print(f"  TN: true FAKE predicted FAKE : {tn}")
    print(f"  FP: true FAKE predicted REAL : {fp}")
    print(f"  FN: true REAL predicted FAKE : {fn}")
    print(f"  TP: true REAL predicted REAL : {tp}")


def analyze_misclassifications(
    X_test: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_examples: int = 3,
):
    """
    Show a few examples of:
      - REAL news flagged as FAKE  (true = 1, pred = 0)
      - FAKE news missed as REAL   (true = 0, pred = 1)
    """
    print_title("STEP 5: ANALYZE FALSE POSITIVES / FALSE NEGATIVES")

    df_eval = pd.DataFrame(
        {
            "text": X_test,
            "true": y_test,
            "pred": y_pred,
            "proba_real": y_proba[:, 1],  # P(class = 1 = REAL)
        },
        index=X_test.index,
    )

    # REAL news flagged as FAKE
    fp_examples = df_eval[(df_eval["true"] == 1) & (df_eval["pred"] == 0)]
    fp_examples = fp_examples.sort_values("proba_real", ascending=False).head(
        n_examples
    )

    # FAKE news missed as REAL
    fn_examples = df_eval[(df_eval["true"] == 0) & (df_eval["pred"] == 1)]
    fn_examples = fn_examples.sort_values("proba_real", ascending=True).head(
        n_examples
    )

    print_subtitle(f"REAL news flagged as FAKE (up to {n_examples})")
    if fp_examples.empty:
        print("No such errors found in this test sample.")
    else:
        for _, row in fp_examples.iterrows():
            print_line("-")
            print(f"Text (truncated): {row['text'][:300].replace('\\n', ' ')}...")
            print("True label      : REAL (1)")
            print("Predicted label : FAKE (0)")
            print(f"P(REAL) = {row['proba_real']:.4f}")

    print_subtitle(f"FAKE news missed as REAL (up to {n_examples})")
    if fn_examples.empty:
        print("No such errors found in this test sample.")
    else:
        for _, row in fn_examples.iterrows():
            print_line("-")
            print(f"Text (truncated): {row['text'][:300].replace('\\n', ' ')}...")
            print("True label      : FAKE (0)")
            print("Predicted label : REAL (1)")
            print(f"P(REAL) = {row['proba_real']:.4f}")


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series):
    """
    Evaluate the trained model:
      - Accuracy
      - Confusion matrix
      - Precision, recall, F1 (classification report)
      - A few misclassified examples
    """
    print_title("STEP 4: EVALUATE MODEL")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Accuracy
    print_subtitle("Accuracy")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)

    # Classification report
    print_subtitle("Classification Report (0 = FAKE, 1 = REAL)")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["FAKE", "REAL"],
            digits=4,
        )
    )

    # Error analysis
    analyze_misclassifications(X_test, y_test, y_pred, y_proba, n_examples=3)


# ----------------------------------------------------------------------
# STEP 6: SIMPLE CLI INTERFACE
# ----------------------------------------------------------------------


def interactive_prediction(model: Pipeline):
    """
    Small loop: type some news text and see if model thinks it's FAKE or REAL.
    """
    print_title("INTERACTIVE MODE: TYPE YOUR OWN NEWS")
    print("Type some news text and press Enter.")
    print("Type 'q' on an empty line to quit.\n")

    while True:
        user_text = input("News text (or 'q' to quit): ").strip()
        if user_text.lower() == "q":
            print("Exiting interactive mode.")
            break

        if not user_text:
            print("Please enter some text.")
            continue

        proba = model.predict_proba([user_text.lower()])[0]
        proba_fake = proba[0]
        proba_real = proba[1]
        pred_label = int(proba_real >= 0.5)
        pred_str = "REAL" if pred_label == 1 else "FAKE"

        print_line("-")
        print(f"Predicted label      : {pred_str}")
        print(f"Probability FAKE (0) : {proba_fake:.4f}")
        print(f"Probability REAL (1) : {proba_real:.4f}")
        print_line("-")
        print()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------


def main():
    # 1) Load and combine data
    df = load_data(FAKE_PATH, REAL_PATH)

    # 2) Prepare X (text) and y (labels)
    X, y = prepare_features_and_labels(df)

    # 3) Train / test split
    print_title("TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train set size: {X_train.shape[0]} rows")
    print(f"Test  set size: {X_test.shape[0]} rows")

    # 4) Build the model pipeline
    model = build_model_pipeline()

    # 5) Train the model
    print_title("TRAINING MODEL")
    model.fit(X_train, y_train)
    print("Training complete.")

    # 6) Evaluate on test data
    evaluate_model(model, X_test, y_test)

    # 7) Optional interactive mode
    try:
        choice = input(
            "\nDo you want to test with your own news text? (y/n): "
        ).strip().lower()
    except EOFError:
        choice = "n"

    if choice == "y":
        interactive_prediction(model)
    else:
        print("\nDone. You can now close the program.")


if __name__ == "__main__":
    main()

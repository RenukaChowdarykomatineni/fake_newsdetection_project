# Fake News Detection

This project builds a machine learning model to classify news articles as **FAKE** or **REAL**.

It uses a Kaggle dataset with two files:

- `Fake.csv` – fake news articles  
- `True.csv` – real news articles  

These files are combined and labelled as:

- `0` = FAKE  
- `1` = REAL  

## How it works

1. Load and combine `Fake.csv` + `True.csv`
2. Create a single text field by combining `title` and `text`
3. Convert text to features using **TF-IDF**
4. Train a **Logistic Regression** classifier
5. Evaluate using:
   - Accuracy
   - Confusion matrix
   - Precision, Recall, F1-score
6. Show some examples of:
   - REAL news flagged as FAKE
   - FAKE news missed as REAL
7. Optional: interactive mode to type your own news text and get a prediction.

## Project structure

```text
fake_news_project/
├── fake_news_detection.py      # Main script
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    ├── Fake.csv                # Fake news (not committed)
    └── True.csv                # Real news (not committed)

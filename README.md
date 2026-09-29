# Email Spam Detector

A Streamlit web application that classifies a message as **spam** or **not spam**. The app loads a trained scikit-learn pipeline and accepts plain-text messages through a simple browser interface.

## Final model

The deployed model is a scikit-learn `Pipeline` containing:

- **Text representation:** `TfidfVectorizer` with unigram and bigram features (`ngram_range=(1, 2)`) and lowercased text
- **Classifier:** `LinearSVC`
- **Tuning:** `GridSearchCV` over `C = [0.01, 0.1, 0.5, 1, 2, 5, 10]`, using 5-fold cross-validation and F1 scoring
- **Selected parameter:** `C=5`

### Evaluation results

Results below are from the held-out test set (1,034 messages) after a stratified 80/20 train/test split with `random_state=42`.

| Metric | Ham (0) | Spam (1) | Overall |
| --- | ---: | ---: | ---: |
| Precision | 0.98 | 0.97 | — |
| Recall | 1.00 | 0.89 | — |
| F1 score | 0.99 | **0.92** | — |
| Accuracy | — | — | **0.98** |

The model's 5-fold cross-validation F1 score during tuning was **0.9442**. Because spam is the class of primary interest, the reported final-model F1 score is the **spam-class F1: 0.92**.

## Dataset and preparation

The project uses `spam.csv`, a labelled SMS/email-style message dataset containing 5,572 rows. The notebook:

1. Removes unused empty columns.
2. Renames the label and message fields to `output` and `email`.
3. Encodes `ham` as `0` and `spam` as `1`.
4. Removes 403 duplicate rows, leaving 5,169 messages.
5. Converts messages to lowercase and performs a stratified 80/20 split.

## Project structure

```text
.
├── app.py                     # Streamlit interface
├── spam_detection_model.pkl   # Saved final TF-IDF + LinearSVC pipeline
├── spam_classifier.ipynb      # Data preparation, experiments, tuning, and evaluation
├── spam.csv                   # Source dataset
├── requirements.txt           # App dependencies
└── IMAGES/                    # Confusion matrices and class-distribution chart
```

## Run locally

Prerequisites: Python 3.9 or later is recommended.

```bash
git clone <your-repository-url>
cd "email spam detector"
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit (usually `http://localhost:8501`), enter a message, and select **Check Message**.

## Reproduce training

Open and run [`spam_classifier.ipynb`](spam_classifier.ipynb) from top to bottom. It compares Multinomial Naive Bayes, Logistic Regression, and Linear SVC, tunes the Linear SVC with cross-validation, evaluates the best estimator on the untouched test set, and saves it as `spam_detection_model.pkl`.

The notebook additionally requires `pandas`, `matplotlib`, and `seaborn` for data preparation and visualisation.

## Limitations

- The data consists of short, labelled messages; performance may differ on modern long-form emails, multilingual content, or adversarially crafted spam.
- The classifier predicts a label only; it does not provide calibrated spam probabilities.
- Retraining is advisable when the intended message source or language changes.

## License

No license has been specified for this repository. Add one before distributing or reusing the project.

# Cuisine Classification

A baseline multi-class classifier that predicts a restaurant's **primary cuisine** from its listing attributes (city, cost, rating, votes, services offered, etc.).

The script is a single file, `cuisine_classification.py`, and runs end-to-end: load data → clean → feature-engineer → split → scale → train a logistic regression → print metrics and plot a confusion matrix.

---

## What it does

1. **Loads** a restaurant CSV (Zomato-style export) and prints a structural summary.
2. **Cleans**: drops rows with no `Cuisines` value, then derives a single-label target — the *first* cuisine listed for each restaurant (e.g. `"North Indian, Chinese"` → `North Indian`).
3. **Reduces the label space**: keeps only the **10 most frequent** primary cuisines and discards the rest, so the model is not swamped by rare classes.
4. **Encodes** features: all remaining categorical columns become one-hot dummy variables; the first level of each is dropped to avoid redundancy.
5. **Splits** 80/20 into train and test sets, stratified on the target so class proportions are preserved in both.
6. **Scales** features with `StandardScaler` — fitted on the training set only, then applied to test, which keeps the test data unseen.
7. **Trains** a `LogisticRegression` with `class_weight="balanced"` to counter class imbalance.
8. **Evaluates** with accuracy, macro and weighted precision/recall, and a full per-class classification report.
9. **Plots** a row-normalised confusion matrix so each cuisine's errors are visible in proportion.

---

## Requirements

```
python >= 3.9
pandas
scikit-learn
matplotlib
```

Install with:

```bash
pip install pandas scikit-learn matplotlib
```

---

## Dataset

The script expects the CSV at a fixed relative location:

```
<project root>/
├── Task_and_dataset/
│   └── Dataset .csv          # note the space before ".csv"
└── <folder containing this script>/
    └── cuisine_classification.py
```

It resolves this via:

```python
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Task_and_dataset" / "Dataset .csv"
```

So the script must sit **one level below** the folder holding `Task_and_dataset`. If your layout differs, edit `DATA_PATH` — that is the only path you should need to touch.

### Columns the data must contain

| Column | Used as |
|---|---|
| `Cuisines` | Source of the target label |
| `Restaurant ID`, `Restaurant Name`, `Address`, `Locality`, `Locality Verbose` | Dropped — identifiers, not signal |
| `Switch to order menu` | Dropped — constant in the source data |
| `Rating color`, `Rating text` | Dropped — they are a direct encoding of `Aggregate rating`, so keeping them leaks the answer |
| Everything else | Kept as a feature |

---

## How to run

```bash
cd <folder containing cuisine_classification.py>
python cuisine_classification.py
```

The confusion matrix opens in a Matplotlib window via `plt.show()`. On a headless machine (server, Docker, CI), either run under a virtual display or replace the last two lines with a save:

```python
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
```

---

## Output

The script writes nothing to disk; everything goes to stdout and the plot window.

```
Accuracy:  0.xx

Macro precision:    0.xx
Macro recall:       0.xx

Weighted precision: 0.xx
Weighted recall:    0.xx

Classification_Report:
              precision  recall  f1-score  support
       Bakery      ...
      Chinese      ...
```

### Reading the metrics

- **Accuracy** — overall share of correct predictions. Misleading on its own here, because the top cuisines are heavily uneven in frequency.
- **Macro precision / recall** — the unweighted mean across the 10 classes. Every cuisine counts equally, so a rare class that the model ignores drags these down. This is the honest headline number for this problem.
- **Weighted precision / recall** — averaged in proportion to class size. Closer to accuracy; flatters the majority classes.
- **Confusion matrix (normalised)** — each row sums to 1.0, so a cell shows *the fraction of true class X predicted as Y*. Off-diagonal mass tells you which cuisines the model confuses, which is usually the most actionable part of the output.

---

## Design notes and trade-offs

**First cuisine only.** `Cuisines` is multi-label in the raw data, but the script collapses it to the first entry. That makes the problem a clean single-label classification, at the cost of throwing away real information — a restaurant tagged `"Cafe, Italian, Desserts"` is counted as a Cafe and nothing else. A multi-label approach (one-vs-rest over a `MultiLabelBinarizer`) would use that information but is a larger change.

**Top-10 classes.** Classes outside the top 10 are dropped rather than grouped into an "Other" bucket. This keeps the task well-posed and the matrix readable; the trade-off is that the model cannot recognise any cuisine outside that set, and if you deploy it, roughly the tail of your catalogue is unclassifiable. An `"Other"` class is the usual compromise.

**Scaling after one-hot encoding.** `StandardScaler` on dummy columns is harmless (it only rescales binary 0/1 columns) and helps the continuous features — `Average Cost for two`, `Aggregate rating`, `Votes` — sit on comparable footing. Note that the scaler is returned as a NumPy array by `fit_transform`, so the DataFrame column names are lost at that point. That is fine for this model but would break anything downstream that expects named columns.

**Leakage avoided.** The scaler is fitted on `X_train` and only *applied* to `X_test`, and the split happens before scaling. Both matter — fitting on the full dataset would leak test statistics into training and inflate the scores.

**Reproducibility.** `random_state=42` fixes the split, so repeated runs are identical.

---

## Known limitations

- **Weak signal.** City, cost band, and rating explain only part of what a restaurant serves; cuisine is largely latent in the name and menu, which are dropped. Expect modest accuracy — this is a baseline, not a ceiling.
- **Linearity.** Logistic regression draws linear decision boundaries in feature space. Cuisine boundaries are not linear; tree-based models (Random Forest, Gradient Boosting, XGBoost) typically beat it substantially on this kind of tabular data and are the natural next step.
- **Class imbalance.** `class_weight="balanced"` partially compensates, but the smallest classes still get few examples to learn from.
- **No hyperparameter tuning.** `max_iter=1000` is raised from the default purely to let the solver converge on scaled data; `C` is left at its default of 1.0.
- **No persisted model.** Nothing is saved, so the script must be re-run to make predictions.
- **No cross-validation.** A single 80/20 split means the reported numbers carry split-to-split variance. `cross_val_score` with 5 folds would give a more stable estimate.

---

## Possible next steps

1. Swap in a tree-based model and compare the same metrics.
2. Add an `"Other"` class instead of dropping the long tail.
3. Tune `C` and the solver with `GridSearchCV`.
4. Use `MultiLabelBinarizer` to predict all listed cuisines, scored with Hamming loss or micro-F1.
5. Pull features from `Restaurant Name` (TF-IDF over the name text) — a surprisingly strong signal for cuisine.
6. Persist the fitted scaler and model with `joblib` so the script can score new rows without retraining.

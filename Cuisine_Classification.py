import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#------------------------------------
# Load data + Drop rows with missing
# values + Single label target: first
# cuisine, keep top N
#------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Task_and_dataset" / "Dataset .csv"

df = pd.read_csv(DATA_PATH)
print(df.info())
df = df.dropna(subset=["Cuisines"])
print(df.isnull().sum())
df["Primary"] = df["Cuisines"].str.split(",").str[0].str.strip()
top = df["Primary"].value_counts().nlargest(10).index
df = df[df["Primary"].isin(top)]

#---------------------------------------
# Features , Target, Encode Categoricals
#---------------------------------------

drop_cols = ["Restaurant ID", "Locality", "Restaurant Name", "Address",
             "Locality Verbose", "Switch to order menu", "Cuisines",
             "Rating color", "Rating text"]

X = df.drop(columns=drop_cols + ["Primary"])
y = df["Primary"]

X = pd.get_dummies(X, drop_first=True)

#--------------
# Split, Scale
#--------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#---------------------
# Train + Evaluate
#---------------------

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
y_pred =model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))

print("\nMacro precision:   ", precision_score(y_test, y_pred, average="macro", zero_division=0))
print("Macro recall:      ", recall_score(y_test, y_pred, average="macro", zero_division=0))

print("\nWeighted precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Weighted recall:   ", recall_score(y_test, y_pred, average="weighted", zero_division=0))

print("\nClassification_Report:\n", classification_report(y_test, y_pred, zero_division=0))

#-------------------------------------------
# Plot performance across different cuisines
#-------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation=45, cmap="Blues", normalize="true")
plt.title("Logistic Regression - Confusion Matrix(normalized)")
plt.tight_layout()
plt.show()

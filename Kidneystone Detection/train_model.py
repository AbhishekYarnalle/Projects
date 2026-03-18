import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_PATH = "dataset"

X = []
y = []

print("Loading dataset...")

# ---------------- LOAD IMAGES ----------------
for label_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_name)

    if os.path.isdir(folder_path):

        label = 0 if label_name.lower() == "normal" else 1

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                img = Image.open(img_path).convert("L")
                img = img.resize((50, 50))
                img_array = np.array(img).flatten() / 255.0

                X.append(img_array)
                y.append(label)

            except:
                print("Error loading:", img_path)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded Successfully!")
print("Total Samples:", len(X))

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- RANDOM FOREST ----------------
print("Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "kidney_rf_model.pkl")

print("\nModel Saved as kidney_rf_model.pkl")

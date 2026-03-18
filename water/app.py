from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")   # ⭐ important
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = "supersecretkey"

# ================= DATABASE =================

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  email TEXT,
                  password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  date TEXT,
                  prediction TEXT)''')

    conn.commit()
    conn.close()

init_db()

# ================= MODEL TRAINING =================

def train_model():
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.mean(), inplace=True)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Model Accuracy:", accuracy)

    pickle.dump(model, open("model.pkl", "wb"))

train_model()
model = pickle.load(open("model.pkl", "rb"))

# ================= ROUTES =================

# -------- HOME (Before Login) --------
@app.route('/')
def index():
    if "username" in session:
        return render_template("index.html")
    return render_template("home.html")


@app.route('/about')
def about():
    if "username" in session:
        return render_template("about.html")
    return render_template("home.html")

# -------- REGISTER --------
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                  (username, email, password))
        conn.commit()
        conn.close()

        df = pd.DataFrame([[username, email, password]],
                          columns=["Username", "Email", "Password"])
        if os.path.exists("users.xlsx"):
            old = pd.read_excel("users.xlsx")
            df = pd.concat([old, df])
        df.to_excel("users.xlsx", index=False)

        flash("Registration Successful!")
        return redirect(url_for('login'))

    return render_template("register.html")


# -------- LOGIN --------
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid Credentials")

    return render_template("login.html")


# -------- LOGOUT --------
@app.route('/logout')
def logout():
    session.pop("username", None)
    flash("Logged out successfully!")
    return redirect(url_for("index"))


# -------- FORGOT --------
@app.route('/forgot', methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form["email"]
        flash("Password recovery feature simulated!")
    return render_template("forgot.html")


# -------- PREDICT --------
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        features = [float(request.form[x]) for x in request.form]
        prediction = model.predict([features])[0]

        result = "Safe Water" if prediction == 1 else "Not Safe"

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO predictions (username,date,prediction) VALUES (?,?,?)",
                  (session["username"], datetime.now(), result))
        conn.commit()
        conn.close()

        df = pd.DataFrame([[session["username"],
                            datetime.now(), result]],
                          columns=["Username", "Date", "Prediction"])

        if os.path.exists("predictions.xlsx"):
            old = pd.read_excel("predictions.xlsx")
            df = pd.concat([old, df])

        df.to_excel("predictions.xlsx", index=False)

        return render_template("predict.html", prediction=result)

    return render_template("predict.html")


# -------- ANALYSIS --------
@app.route('/analysis')
def analysis():
    if "username" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv("water_potability.csv")
    df.fillna(df.mean(), inplace=True)

    plt.figure()
    df["Potability"].value_counts().plot(kind='bar')
    plt.title("Water Potability Distribution")
    plt.savefig("static/bar.png")
    plt.close()

    plt.figure()
    df.hist(figsize=(10,8))
    plt.savefig("static/hist.png")
    plt.close()

    return render_template("analysis.html")


# -------- SOLUTION --------
@app.route('/solution')
def solution():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("solution.html")


# -------- CONTACT --------
@app.route('/contact', methods=["GET", "POST"])
def contact():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name")
        message = request.form.get("message")
        flash("Message sent successfully!")

    return render_template("contact.html")




# -------- VISUALIZATION --------
@app.route('/vis')
def vis():
    if "username" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv("water_potability.csv")
    df.fillna(df.mean(), inplace=True)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ================= MODEL COMPARISON =================
    rf = RandomForestClassifier(n_estimators=200)
    lr = LogisticRegression(max_iter=200)

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    plt.figure()
    plt.bar(["RandomForest","Logistic"], [rf_acc, lr_acc])
    plt.title("Model Comparison")
    plt.savefig("static/model_compare.png")
    plt.close()

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_test, rf.predict(X_test))

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("static/confusion.png")
    plt.close()

    # ================= FEATURE IMPORTANCE =================
    plt.figure(figsize=(8,5))
    importances = rf.feature_importances_
    plt.barh(X.columns, importances)
    plt.title("Feature Importance")
    plt.savefig("static/feature.png")
    plt.close()

    # ================= CONFIDENCE =================
    probs = rf.predict_proba(X_test)[:,1][:100]

    plt.figure()
    plt.plot(probs)
    plt.title("Prediction Confidence")
    plt.savefig("static/confidence.png")
    plt.close()

    # ================= OUTLIER DETECTION =================
    z = (df - df.mean())/df.std()

    plt.figure()
    plt.hist(z["Solids"], bins=40)
    plt.title("Outlier Detection (Solids)")
    plt.savefig("static/outlier.png")
    plt.close()

    # ================= DISTRIBUTION =================
    plt.figure()
    df["Potability"].value_counts().plot(kind="bar")
    plt.title("Dataset Distribution")
    plt.savefig("static/distribution.png")
    plt.close()

    return render_template("vis.html")


# -------- PROJECT INFO --------
@app.route('/info')
def info():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("project_info.html")




if __name__ == "__main__":
    app.run(debug=True)


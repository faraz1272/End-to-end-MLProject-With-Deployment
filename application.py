from flask import Flask, request, render_template, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import os
import joblib
import csv, datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])
    
@app.get("/health")
def health():
    return {"status": "ok"}, 200

FEATURE_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score",
]

def _payload_to_df(payload: dict) -> pd.DataFrame:
    row = {
        "gender": payload.get("gender"),
        "race_ethnicity": payload.get("race_ethnicity") or payload.get("ethnicity"),
        "parental_level_of_education": payload.get("parental_level_of_education"),
        "lunch": payload.get("lunch"),
        "test_preparation_course": payload.get("test_preparation_course"),
        "reading_score": float(payload.get("reading_score")),
        "writing_score": float(payload.get("writing_score")),
    }
    # basic validation
    missing = [c for c, v in row.items() if v is None]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

@app.post("/predict")
def predict_api():
    """
    JSON endpoint for programmatic predictions.

    Example body:
    {
      "gender": "female",
      "race_ethnicity": "group B",
      "parental_level_of_education": "bachelor's degree",
      "lunch": "standard",
      "test_preparation_course": "none",
      "reading_score": 72,
      "writing_score": 70
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        df = _payload_to_df(payload)
        preds = PredictPipeline().predict(df)
        return jsonify({"prediction": float(preds[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

    
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
TARGET_COL = "math_score"

@app.route("/api/dashboard-data", methods=["GET"])
def dashboard_data():
    test_csv = os.path.join(ARTIFACTS_DIR, "test.csv")
    df = pd.read_csv(test_csv)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].to_numpy()

    preds = PredictPipeline().predict(X)
    preds = np.asarray(preds).reshape(-1)

    r2 = float(r2_score(y, preds))
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    limit = min(200, len(df))
    points = [{"actual": float(y[i]), "pred": float(preds[i])} for i in range(limit)]
    residuals = [float(preds[i] - y[i]) for i in range(limit)]

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    model_name = None
    try:
        model_obj = joblib.load(model_path)
        model_name = type(model_obj).__name__
    except Exception:
        model_name = "Unknown"

    return jsonify({
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "points": points,
        "residuals": residuals,
        "model_name": model_name               # <— added
    })

@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    """
    Permutation importance on a small sample of test.csv.
    Returns: [{feature, delta_rmse}], sorted desc.
    """
    test_csv = os.path.join(ARTIFACTS_DIR, "test.csv")
    df = pd.read_csv(test_csv)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].to_numpy()

    n = min(500, len(X))
    X = X.sample(n=n, random_state=42)
    y = y[X.index]

    pp = PredictPipeline()

    # Baseline RMSE
    base_pred = pp.predict(X)
    base_rmse = float(np.sqrt(mean_squared_error(y, base_pred)))

    importances = []
    rng = np.random.default_rng(42)

    for col in FEATURE_COLUMNS:
        X_shuffled = X.copy()

        X_shuffled[col] = X_shuffled[col].sample(frac=1.0, random_state=42).values

        pred = pp.predict(X_shuffled)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        importances.append({"feature": col, "delta_rmse": rmse - base_rmse})

    # sort desc, keep top 10
    importances.sort(key=lambda d: d["delta_rmse"], reverse=True)
    importances = importances[:10]

    return jsonify({
        "baseline_rmse": base_rmse,
        "importances": importances
    })

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/eda", methods=["GET"])
def eda_summary():
    """EDA for landing page cards & charts."""
    csv_path = os.path.join(ARTIFACTS_DIR, "train.csv")
    df = pd.read_csv(csv_path)

    target = "math_score"
    num_cols = ["reading_score", "writing_score"]

    # Basic stats
    n = int(len(df))
    mean = float(df[target].mean())
    median = float(df[target].median())
    std = float(df[target].std(ddof=1))

    # Correlations (with numeric columns)
    corr = {}
    for c in num_cols:
        if c in df.columns:
            corr[c] = float(df[target].corr(df[c]))

    # Histogram for target
    counts, edges = np.histogram(df[target].dropna(), bins=12)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    # Simple categorical breakdowns
    def group_mean(col):
        if col in df.columns:
            m = df.groupby(col)[target].mean().sort_values(ascending=False)
            return [{"label": str(k), "value": float(v)} for k, v in m.items()]
        return []
    
    insights = []
    by_gender = group_mean("gender")
    by_lunch = group_mean("lunch")
    by_prep = group_mean("test_preparation_course")
    by_parent = group_mean("parental_level_of_education")
    by_race = group_mean("race_ethnicity")

    # gender: top vs bottom
    if len(by_gender) >= 2:
        g = sorted(by_gender, key=lambda x: x["value"], reverse=True)
        delta = g[0]["value"] - g[-1]["value"]
        insights.append(
            f"{g[0]['label'].title()} students score {delta:.1f} points higher on average than {g[-1]['label'].title()}."
        )

    # lunch: standard vs free/reduced if present
    if len(by_lunch) >= 2:
        l = sorted(by_lunch, key=lambda x: x["value"], reverse=True)
        delta = l[0]["value"] - l[-1]["value"]
        insights.append(
            f"Students with '{l[0]['label']}' lunch average {delta:.1f} points higher than '{l[-1]['label']}'."
        )

    # test prep: completed vs none
    if len(by_prep) >= 2:
        p = sorted(by_prep, key=lambda x: x["value"], reverse=True)
        delta = p[0]["value"] - p[-1]["value"]
        insights.append(
            f"Those who '{p[0]['label']}' score {delta:.1f} points higher than those who '{p[-1]['label']}'."
        )

    # parental education: highest vs lowest
    if len(by_parent) >= 2:
        pe = sorted(by_parent, key=lambda x: x["value"], reverse=True)
        insights.append(
            f"Parental education: '{pe[0]['label']}' shows the highest average ({pe[0]['value']:.1f}) vs '{pe[-1]['label']}' ({pe[-1]['value']:.1f})."
        )

    # race/ethnicity: top group mention
    if len(by_race) >= 2:
        r = sorted(by_race, key=lambda x: x["value"], reverse=True)
        insights.append(
            f"Top group by average math score: {r[0]['label']} ({r[0]['value']:.1f})."
        )

    # correlations note
    if "reading_score" in corr and "writing_score" in corr:
        insights.append(
            f"Correlations with math: reading {corr['reading_score']:.2f}, writing {corr['writing_score']:.2f}."
        )

    return jsonify({
    "n": n,
    "mean": mean,
    "median": median,
    "std": std,
    "corr": corr,
    "hist": {"centers": centers, "counts": counts.tolist()},
    "by_gender": by_gender,
    "by_lunch": by_lunch,
    "insights": insights
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, "feedback.csv")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["ts_utc", "name", "email", "message"])
        w.writerow([datetime.datetime.utcnow().isoformat(), name, email, message])

    return redirect(url_for("index", feedback="thanks"))
    
if __name__ == "__main__":
    app.run(port=5001)
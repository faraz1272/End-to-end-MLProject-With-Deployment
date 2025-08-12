from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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

# Accept both "race_ethnicity" (API) and "ethnicity" (form) for convenience
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
    
if __name__ == "__main__":
    app.run(port=5001)
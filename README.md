# Student Exam Score Predictor

> **What:** A production-ready ML web app that predicts a student's **math score** from demographics and prior scores and surfaces **explainable insights** via an interactive dashboard.
>
> **Why:** Help educators and analysts quickly spot patterns that correlate with performance and experiment with "what-if" inputs.
>
> **Live demo:** [http://student-score-api-env.eba-3nhfdau5.eu-west-2.elasticbeanstalk.com](http://student-score-api-env.eba-3nhfdau5.eu-west-2.elasticbeanstalk.com)

## Table of Contents
- [Key Insights from EDA](#key-insights-from-eda)
- [Goals & Scope](#goals--scope)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data & Features](#data--features)
- [Modeling](#modeling)
- [Experiment Tracking](#experiment-tracking)
- [Local Setup](#local-setup)
- [Docker](#docker)
- [API & UI](#api--ui)
- [CI/CD](#cicd)
- [AWS Deployment](#aws-deployment)
- [Security Notes](#security-notes)
- [Roadmap](#roadmap)
- [Quick Start](#quick-start)
- [License & Credits](#license--credits)

## Key Insights from EDA

Below are key insights discovered in the notebook EDA and reproduced in the landing page dashboard:

- **Score distribution**: Math/Reading/Writing scores center around **66–68** with standard deviations around **~15**; math has the lowest minima (including **0**), indicating heavier lower tail.
- **Subject difficulty**: Students generally **underperform in Math** compared to Reading/Writing (most low-end outliers come from Math).
- **Gender**: **Females have higher overall averages**, while **males edge Math** slightly in some views.
- **Lunch**: **Standard lunch** students consistently score higher than **free/reduced** lunch—visible across genders.
- **Parental education**: Mixed overall effect; **Bachelor's/Master's** tends to associate with higher averages; effect varies by gender subgroup.
- **Race/Ethnicity**: Group **E** has the highest mean across subjects; Group **A** the lowest in this dataset.
- **Test preparation**: **Completed** prep course → higher scores in all three subjects vs **None**.
- **Relationships**: Scores in Math/Reading/Writing show **strong positive linear relationships** (pairplots/correlations).

### EDA Visualizations

<!-- Replace with your actual image paths -->
![Score Histograms](images(plots)/output.png)
*Distribution of Math, Reading, and Writing scores*

![Average Scores by Lunch & Gender](images(plots)/output2.png)
*Performance comparison across lunch types and gender*

![Scores by Parental Education](images(plots)/impact_of_parent_edu.png)
*Impact of parental education level on student performance*

![Group-wise Subject Means](images(plots)/score_by_ethnicity.png)
*Average scores by race/ethnicity groups across subjects*

![Score Correlations](images(plots)/corr.png)
*Correlation matrix and pairplot of all three subjects*

## Goals & Scope

**Goal**: Estimate a student's math score using easily available attributes, and present clear, **actionable** patterns that support interventions (e.g., lunch programs, prep courses).

**Non-goals**: High-stakes decision making or causal inference; this is a **predictive** model over a single dataset with known biases.

### Objectives
- Provide a clean **inference API** and **UI** for exploration & predictions
- Track experiments, metrics, and artifacts with **MLflow**
- Ship with **Docker** + **CI/CD** to AWS for reliable, repeatable deploys

## Architecture

```
Data (stud.csv) ─► Ingestion ─► Transformation (preprocessor.pkl) ─► Training (model.pkl)
                                          │
                                          ├─► MLflow (params, metrics, artifacts)
                                          │
                                  ┌───────▼─────────────────────────────────┐
                                  │    Flask + Gunicorn (API + UI)          │
                                  │  - /predict (JSON)                      │
                                  │  - / (Landing with EDA KPIs)            │
                                  │  - /dashboard (Metrics & charts)        │
                                  └───────▲─────────────────────────────────┘
                                          │
                                   Docker container
                                          │
                            GitHub Actions  (CI/CD pipeline)
                                          │
                            Amazon ECR  ──► Elastic Beanstalk (deploy)
```

## Repository Structure

```
.
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── utils.py
│   ├── logger.py
│   ├── exception.py
│   └── __init__.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   └── home.html
├── application.py                # Flask app (API + UI + JSON endpoints)
├── artifacts/                    # model.pkl, preprocessor.pkl, train.csv, test.csv
├── Dockerfile.api                # production image (Gunicorn)
├── Dockerrun.aws.json            # EB single-container Docker descriptor
├── requirements.txt
├── .github/workflows/ci.yml      # CI/CD: build, ECR push, EB deploy
└── README.md
```

## Data & Features

- **Dataset**: *Students Performance in Exams* (Kaggle). 8 columns × 1000 rows
- **Target**: `math_score`
- **Inputs**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`

### Transformations
- **Numerical** (`reading_score`, `writing_score`): median imputation + StandardScaler
- **Categorical** (the rest): most-frequent imputation + One-Hot + StandardScaler(with_mean=False)
- Saved to `artifacts/preprocessor.pkl`

## Modeling

- **Models benchmarked**: RandomForest, DecisionTree, GradientBoosting, LinearRegression, KNN, XGBoost, CatBoost, AdaBoost
- **Hyperparameters**: grid/param search per model
- **Metric for model selection**: **R²** on hold-out; additional **MAE/RMSE** reported on dashboard
- **Best model** is serialized to `artifacts/model.pkl`

### Feature Importance
**Permutation importance** using **ΔRMSE**: for each feature, shuffle values and measure RMSE increase vs baseline. Higher Δ ⇒ more important. (Correlated features can share credit.)

## Experiment Tracking

MLflow runs record parameters, metrics (R²/MAE/RMSE), and artifacts. Config-driven training lets you reproduce runs and compare models.

> **Tip**: Start an MLflow server locally or log to the local `mlruns/` directory. The training entrypoint accepts a config path.

## Local Setup

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train & Create Artifacts

```bash
# Example config path; adapt to your repo
python -m src.pipeline.train_pipeline --config configs/train.yaml
# artifacts/model.pkl and artifacts/preprocessor.pkl will be produced
```

### 3. Start the API/UI (Development)

```bash
export FLASK_APP=application.py
export FLASK_ENV=development
python application.py   # or: flask run --port 8080
```

Open: [http://localhost:8080/](http://localhost:8080/)

## Docker

### Local Docker Setup

```bash
# Build
docker build -f Dockerfile.api -t student-score-api:local .

# Run
docker run --rm -p 8080:8080 student-score-api:local
```

Open: [http://localhost:8080/](http://localhost:8080/)

### Environment Variables
- `ARTIFACTS_DIR` (default: `artifacts`) – path inside the container from which the app loads model/preprocessor and CSVs

## API & UI

### Health Check
```http
GET /health
```
**Response**: `{"status":"ok"}`

### Prediction API
```http
POST /predict
Content-Type: application/json

{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "none",
  "reading_score": 72,
  "writing_score": 70
}
```
**Response**: `{ "prediction": 63.21 }`

### UI Routes
- `/` – Landing page with KPIs, EDA charts, insights, project overview, and feedback form
- `/predictdata` – Form UI for predictions
- `/dashboard` – Metrics (R²/MAE/RMSE), Pred vs Actual, Residuals, and Permutation Importance

### JSON Data Endpoints
- `/api/eda` – Summary stats, histogram bins, group means, and auto-generated insights
- `/api/dashboard-data` – Hold-out metrics + sampled pred/actual points + residuals + model name
- `/api/feature-importance?metric=rmse|mae|r2` – Permutation importance by chosen metric

## CI/CD

### Continuous Integration (GitHub Actions)

**On every push/PR:**
1. **Build & Smoke Test API** – Build Docker image, run container, verify `/health`

**On push to `main`:**
2. **Build & Push to Amazon ECR** – Push `:latest` and `:<commit-sha>` tags
3. **Deploy to Elastic Beanstalk** – Create and deploy new application version

### AWS Prerequisites
- **ECR repository**: `student-score-api`
- **Elastic Beanstalk**: Application & Single-instance Docker environment
- **S3 artifacts bucket**: e.g., `elasticbeanstalk-<region>-<account>`
- **IAM (OIDC for GitHub)**: Role with ECR, S3, EB, and CloudFormation permissions
- **Instance profile**: `AmazonEC2ContainerRegistryReadOnly` for ECR image pulls

### Rollbacks
EB → *Application versions* → select previous version → **Deploy**

## AWS Deployment

- Image pulled from **Amazon ECR** via `Dockerrun.aws.json`
- Health check path: `/health`
- Container port: **8080**
- Single-instance environment exposes **HTTP :80**

### HTTPS Options
- **Quick**: CloudFront in front (Origin = EB, Policy = *Redirect HTTP to HTTPS*)
- **Full**: Load Balanced EB + ACM cert + ALB :443 listener

## Security Notes

⚠️ **Important**: This project is for **educational** use; dataset biases propagate into predictions. Do not use for high-stakes decisions without rigorous validation and governance.

## Roadmap

- [ ] Move artifacts to **S3** + optional MLflow Model Registry
- [ ] Add **SHAP** for local explanations
- [ ] Add user authentication & per-session input history
- [ ] Add CloudFront HTTPS by default
- [ ] Implement model versioning and A/B testing

## Quick Start

```bash
# Local quickstart
pip install -r requirements.txt && python application.py
# Visit http://localhost:8080/

# Docker quickstart
docker build -f Dockerfile.api -t student-score-api:local .
docker run --rm -p 8080:8080 student-score-api:local

# CI/CD: push to main to trigger ECR push + EB deploy
git push origin main
```

## License & Credits

- **Data**: *Students Performance in Exams* (Kaggle)
- **Libraries**: Python, scikit-learn, XGBoost, CatBoost, Flask, Gunicorn, Chart.js, MLflow
- **Infrastructure**: Docker, GitHub Actions, Amazon ECR, Elastic Beanstalk, S3, CloudWatch

---

> 💡 **Questions, bugs, or ideas?** Open an issue or use the Feedback form on the landing page.
# 🍽️ AI Demand Forecasting & Inventory Optimization

> **Infotact Technical Internship Program — Project 3: Food & Restaurant Services**

An AI-powered demand forecasting system built for simulated restaurant and food delivery chains. Uses historical point-of-sale (POS) data alongside engineered time-series features to predict daily sales volume — enabling smarter inventory decisions, reduced food waste, and optimized labor scheduling.

---

## 📌 Problem Statement

The restaurant industry operates on razor-thin margins with highly perishable inventory. Traditional planning methods (spreadsheets, gut feeling, simple averages) fail to capture the dynamic patterns driving customer demand — leading to costly over-ordering or revenue-losing stockouts.

This project builds a data-driven forecasting engine that predicts next-day demand by learning from historical trends, seasonal patterns, and engineered features.

---

> Accurate forecasting is estimated to drive a **20–30% reduction** in food waste and warehousing costs.

---

## 🗂️ Project Structure

```
restaurant-demand-forecasting/
│
├── data/                        # Raw and processed datasets (gitignored)
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── w1_time_series_eda.ipynb         # Week 1: EDA & data ingestion
│   ├── w2_feature_engineering.ipynb     # Week 2: Feature engineering
│   ├── w3_model_training.ipynb          # Week 3: Model training & selection
│   └── week4_complete_notebook.ipynb    # Week 4: Evaluation & business reporting
│
├── models/                      # Saved model weights (gitignored)
│   └── best_model_xgb.pkl
│
├── src/
│   ├── preprocessing.py         # Data wrangling utilities
│   ├── feature_engineering.py   # Feature creation pipeline
│   └── evaluate.py              # MAE, RMSE evaluation helpers
│
├── reports/
│   └── forecast_vs_actuals.png  # Final visualized predictions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Four-Week Engineering Roadmap

### Week 1 — Data Ingestion & Time-Series EDA
- Acquired and cleaned restaurant/retail sales dataset
- Formatted datetime index for continuity
- Plotted overall sales trend and seasonal decomposition
- Analyzed autocorrelations to understand demand persistence

### Week 2 — Advanced Feature Engineering
- Engineered chronological features: `day_of_week`, `month`, `is_weekend`, `is_holiday`
- Created **lag features** (e.g., sales 7 days ago) to encode temporal memory
- Built **rolling window statistics** (e.g., 14-day moving average)
- Performed time-aware train/test split (first 10 months → train, last 2 → test) to prevent data leakage

### Week 3 — Model Training & Selection
- Established baseline with **Linear Regression** (MAE: 438.92, RMSE: 716.25)
- Trained **Random Forest** (MAE: 388.57, RMSE: 588.65)
- Trained **XGBoost** (MAE: 378.28, RMSE: 611.25) ← Best MAE
- Performed hyperparameter tuning with **TimeSeriesSplit cross-validation**

### Week 4 — Evaluation, Feature Importance & Business Reporting
- Visualized predictions vs. actuals across the test set
- Extracted feature importance scores to explain demand drivers
- Documented business insights and model recommendations

---

## 📊 Model Comparison

| Model | MAE | RMSE |
|---|---|---|
| **XGBoost** | **378.28** | 611.25 |
| Random Forest | 388.57 | 588.65 |
| Linear Regression | 438.92 | 716.25 |

> **Winner: XGBoost** — lowest MAE, making it the best model for minimizing average daily prediction error.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Forecasting Models | Scikit-Learn, XGBoost, Prophet |
| Visualization | Matplotlib, Plotly |
| Model Persistence | Joblib |
| Version Control | Git & GitHub |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/keshava8769/Restaurant-demand-forecasting-and-inventory-optimization-Ai.git
cd Restaurant-demand-forecasting-and-inventory-optimization-Ai
```

### 2. Run Notebooks in Order
```
notebooks/w1_time_series_eda.ipynb
notebooks/w2_feature_engineering.ipynb
notebooks/w3_model_training.ipynb
notebooks/week4_complete_notebook.ipynb
```

---

## 🔒 Security & Data Guidelines

- Raw datasets (`.csv`, `.xlsx`) and model weights (`.pkl`) are **gitignored** and never pushed to the repository
- API keys and credentials must **never** be hardcoded — use environment variables instead
- Jupyter notebook output cells are cleared before committing to prevent repository bloat

---

## 📈 Commit Standards

This project follows structured, semantic commit messages:

```
data-clean: removed duplicate dates and fixed datetime index
eda: plotted weekly seasonality and autocorrelation (ACF/PACF)
feature-eng: added lag_7, lag_14 and rolling_mean_14 features
model-tuning: optimized XGBoost depth using TimeSeriesSplit CV
eval: generated forecast vs actuals plot and feature importance chart
```

> Minimum **3–5 meaningful commits per active development day** as per Infotact MLOps standards.

---

## 👤 Author

**Chenna Keshava, Khushali Chavda, Prem Prajapati, Ashish Gupta**
Infotact Technical Internship Program — Data Science & Machine Learning
📍 Bengaluru, Karnataka

---

## 📄 License

This project is developed as part of the Infotact internship program and follows its internal evaluation and submission guidelines.

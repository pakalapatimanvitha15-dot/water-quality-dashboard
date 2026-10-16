# 🌊 HydroIntel: Water Quality Assessment Dashboard
### ML × ACS Hackathon 2026 | Environmental Chemistry & Machine Learning Track

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-teal.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An environmental machine learning screening system and interactive analytical dashboard developed for the **ML × ACS Hackathon 2026**. HydroIntel provides real-time water sample assessment, anomaly detection, physicochemical indicator exploration, and model transparency based on historical monitoring records from the **Central Pollution Control Board (CPCB)**.

---

## 📌 Executive Summary & Key Findings

Before building the dashboard, the pre-existing assets (`water_quality_model.pkl`, `model_features.json`, and `water_dataX.csv`) were systematically inspected to ensure complete scientific integrity:

1. **Pre-trained Model Architecture (`water_quality_model.pkl`):**
   - **Estimator:** `sklearn.ensemble.RandomForestClassifier` with `100` decision trees.
   - **Serialization:** Saved via `joblib` with serialized NumPy arrays.
   - **Hyperparameters:** `class_weight='balanced'`, `random_state=42`, `criterion='gini'`.
   - **Target Labels:** Binary classification predicting `['Outside pH range', 'Within pH range']`.
   - **Expected Input Dimensionality:** Exactly **7 numeric features** (`n_features_in_ = 7`).

2. **Feature Alignment Deep-Dive (`model_features.json` vs. Model):**
   - `model_features.json` lists 8 features: `Temp`, `D.O. (mg/l)`, `PH`, `CONDUCTIVITY (µmhos/cm)`, `B.O.D. (mg/l)`, `NITRATENAN N+ NITRITENANN (mg/l)`, `FECAL COLIFORM (MPN/100ml)`, and `TOTAL COLIFORM (MPN/100ml)Mean`.
   - In the training pipeline, `PH` was utilized to construct the ground-truth classification target: **Within permissible pH range [6.5 – 8.5]** (as defined by WHO and BIS IS 10500 standards) vs. **Outside pH range (<6.5 or >8.5)**.
   - To avoid **data leakage**, `PH` was intentionally dropped from the feature matrix $X$, leaving the remaining **7 physicochemical and microbial parameters** as the exact inputs required by the model.

3. **Dataset Characteristics (`water_dataX.csv`):**
   - Contains **1,991 historical monitoring records** across 203 state-level regions between **2003 and 2014**.
   - Encoded in `Latin-1` (due to the micro symbol `µ` in `CONDUCTIVITY (µmhos/cm)`).
   - Missing values were recorded as string literal `'NAN'`, which the dashboard standardizes to IEEE-754 nulls (`np.nan`).

4. **Model Performance on Available Labelled Data:**
   - In accordance with hackathon rules, **no model metrics or accuracy figures were invented**.
   - Evaluating the pre-trained model empirically across the complete set of **1,577 records** with complete measurements and recorded ground-truth pH reveals:
     - **Overall Accuracy:** `95.43%`
     - **Outside pH Range Precision:** `97.94%` (Recall: `57.58%`, F1: `72.52%`)
     - **Within pH Range Precision:** `95.27%` (Recall: `99.86%`, F1: `97.51%`)

---

## 🎨 Design & Dashboard Highlights

- **Modern Environmental Science Theme:** Deep navy sidebar (`#0d1b2a`), subtle glowing glassmorphic cards (`#1b263b`), and vibrant teal/cyan accents (`#00f5d4`, `#00b4d8`, `#06d6a0`).
- **4 Dedicated Dashboard Views:**
  1. 📊 **Overview & Data Exploration:** Dataset telemetry, missing data profiles, interactive parameter distributions, bivariate scatter plots, correlation heatmaps, and temporal trends (2003–2014).
  2. 🧪 **Water Sample Assessment:** Interactive laboratory input form with the exact 7 features, one-click realistic scenarios, real-time prediction output, probability breakdown gauge, multi-parameter radar chart, and CPCB/WHO threshold compliance.
  3. 🧠 **Model Inspection & Performance:** Full architectural inspection, Gini feature importance chart (Mean Decrease in Impurity), confusion matrix heatmap, and classification report.
  4. 📜 **Guidelines & Chemistry Guide:** CPCB Designated Best Use Classes (A through E), carbonate-bicarbonate buffer equations ($H_2CO_3 \rightleftharpoons HCO_3^- + H^+$), and chemical explanations for why BOD, DO, and coliforms influence pH stability.
- **Mandatory Safety Disclaimers:** Explicit compliance banners emphasize that machine learning screening is **NOT a drinking-water safety certificate** and cannot substitute for ISO/BIS/EPA accredited laboratory testing.

---

## 🗂️ Project Architecture

```
water-quality-dashboard/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation and setup instructions
├── model_features.json         # Original 8-feature metadata JSON
├── water_dataX.csv             # Central Pollution Control Board raw dataset
├── water_quality_model.pkl     # Pre-trained RandomForestClassifier (100 trees)
└── src/                        # Modular architecture package
    ├── __init__.py             # Package initializer
    ├── data_loader.py          # Data ingestion, encoding fallbacks, null cleaning, stats
    ├── model_utils.py          # Model loading, metadata extraction, prediction, metrics
    ├── visualizations.py       # Plotly charts styled with dark navy/teal aesthetic
    └── ui_components.py        # Custom CSS, banners, badges, cards, sample presets
```

---

## 🚀 Setup & Local Installation

### Prerequisites
- Python **3.10** or higher (tested and verified on Python 3.10, 3.11, 3.12, 3.13, and 3.14 on Windows, macOS, and Linux).

### 1. Clone or Open Workspace
```bash
cd water-quality-dashboard
```

### 2. (Recommended) Create a Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard locally:

```bash
streamlit run app.py
```

The application will automatically start and open in your default browser at:
👉 **`http://localhost:8501`**

To specify a custom port or run in headless mode:
```bash
streamlit run app.py --server.port 8501 --server.headless true
```

---

## 🧪 Testing & Verification

### Automated Verification
An automated test script is provided in the repository that tests end-to-end data loading, cleaning, model loading, preset predictions, evaluation metrics calculation, and Plotly figure generation:

```bash
python -u scratch/test_pipeline.py
```

Expected output:
- `1. Testing data loading... Success: 1991 rows loaded.`
- `2. Testing data cleaning... Success: Nulls and types cleaned.`
- `3. Testing model loading and metadata... Success: Model class=RandomForestClassifier, features=7, trees=100`
- `4. Testing feature importances... Success: Top feature is FECAL COLIFORM with 19.36%`
- `5. Testing predictions on presets... Success`
- `6. Testing evaluation on labelled data... Success: Accuracy=95.43%`
- `7. Testing Plotly visualizations generation... Success`

### Manual Verification in the Web Interface
1. **Explore the Overview:** Navigate to `📊 Overview & Exploration`. Verify the summary metrics (1,991 records, 203 state regions, 2003–2014 coverage), missing value bars, and parameter distributions.
2. **Test Scenario Presets:** Navigate to `🧪 Water Sample Assessment`. Click any preset button (e.g. *Pristine Headwater Stream* or *Dataset Anomaly Station*).
3. **Run Assessment:** Click **`🚀 Assess Water Sample Quality`**. Verify the model output badge, probability confidence bar, multi-parameter radar chart, and threshold compliance table.
4. **Inspect Model Metrics:** Navigate to `🧠 Model Inspection & Metrics`. View the Gini feature importance chart and the confusion matrix computed on the 1,577 complete labelled historical samples.

---

## 🔬 Feature Importance Ranking

Derived directly from `model.feature_importances_`:

| Rank | Parameter | Gini Importance | Share (%) | Environmental Role |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Fecal Coliform** | `0.1936` | **19.36%** | Pathogenic indicator; enteric sewage contamination |
| **2** | **Total Coliform** | `0.1899` | **18.99%** | General sanitary bacterial load and organic decay |
| **3** | **Electrical Conductivity** | `0.1677` | **16.77%** | Dissolved mineral ions ($Ca^{2+}, Mg^{2+}, Cl^-$) and ionic strength |
| **4** | **Biochemical Oxygen Demand** | `0.1272` | **12.72%** | Microbial respiration; drives carbonic acid production |
| **5** | **Temperature** | `0.1176` | **11.76%** | Controls gas solubility ($O_2, CO_2$) and reaction kinetics |
| **6** | **Nitrate + Nitrite** | `0.1027` | **10.27%** | Nutrient loading; stimulates eutrophication & algal blooms |
| **7** | **Dissolved Oxygen** | `0.1014` | **10.14%** | Primary indicator of aerobic self-purification capacity |

---

## 🛡️ Critical Safety Notice

> [!WARNING]
> **PRELIMINARY SCREENING ONLY:** This dashboard and machine learning model are developed strictly for research, educational, and preliminary environmental screening purposes in the **ML × ACS Hackathon 2026**.
> 
> Algorithmic predictions **DO NOT constitute a drinking-water safety certificate** or regulatory compliance guarantee. Legally certified drinking water safety requires physical, chemical, radiological, and microbiological assays conducted by an accredited laboratory in accordance with **BIS (IS 10500:2012)** or **WHO Drinking Water Guidelines**.

---

## 👥 Contributors & Acknowledgements
- **Team HydroIntel** — ML × ACS Hackathon 2026
- **Data Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.

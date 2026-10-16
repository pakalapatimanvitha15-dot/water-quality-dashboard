"""
Machine learning model utilities for loading, inspecting, evaluating,
and predicting with the trained Water Quality RandomForestClassifier.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Environmental standards criteria (CPCB / WHO guidelines for surface & potable water)
ENVIRONMENTAL_BENCHMARKS = {
    'Temp': {
        'name': 'Water Temperature',
        'unit': '°C',
        'optimal': (15.0, 30.0),
        'warning': (30.0, 35.0),
        'critical_high': 35.0,
        'description': 'Elevated temperatures decrease dissolved oxygen solubility and increase biological oxygen demand.'
    },
    'D.O. (mg/l)': {
        'name': 'Dissolved Oxygen (D.O.)',
        'unit': 'mg/L',
        'optimal_min': 6.0,
        'standard_min': 4.0,  # Minimum required to sustain healthy fish & aerobic aquatic ecology
        'critical_low': 2.0,
        'description': 'Essential for aerobic organisms; levels below 4 mg/L cause ecological distress or septic conditions.'
    },
    'CONDUCTIVITY (\u00b5mhos/cm)': {
        'name': 'Electrical Conductivity',
        'unit': 'µmhos/cm',
        'optimal_max': 500.0,
        'standard_max': 1000.0,
        'critical_high': 2000.0,
        'description': 'Measures ionic mineral content; sudden spikes indicate dissolved solids, saline intrusion, or chemical runoff.'
    },
    'B.O.D. (mg/l)': {
        'name': 'Biochemical Oxygen Demand (B.O.D.)',
        'unit': 'mg/L',
        'optimal_max': 2.0,
        'standard_max': 3.0,  # Standard for outdoor bathing / clean surface streams
        'critical_high': 8.0,
        'description': 'Oxygen required by microbes to decompose organic matter; high values signify raw sewage or industrial organic pollution.'
    },
    'NITRATENAN N+ NITRITENANN (mg/l)': {
        'name': 'Nitrate + Nitrite Nitrogen',
        'unit': 'mg/L',
        'optimal_max': 1.0,
        'standard_max': 10.0,  # WHO drinking threshold is 10 mg/L as NO3-N (or 45-50 mg/L as NO3)
        'critical_high': 20.0,
        'description': 'Nutrient indicator from agricultural fertilizers and human waste; promotes toxic algal blooms and eutrophication.'
    },
    'FECAL COLIFORM (MPN/100ml)': {
        'name': 'Fecal Coliform Count',
        'unit': 'MPN/100 mL',
        'optimal_max': 50.0,
        'standard_max': 500.0,  # CPCB outdoor bathing desirable limit: 500 MPN/100ml
        'critical_high': 2500.0,
        'description': 'Pathogenic bacteria originating from human/animal intestinal tracts; indicates direct sewage contamination.'
    },
    'TOTAL COLIFORM (MPN/100ml)Mean': {
        'name': 'Total Coliform Count',
        'unit': 'MPN/100 mL',
        'optimal_max': 500.0,
        'standard_max': 5000.0,  # CPCB standard for drinking water with conventional treatment
        'critical_high': 10000.0,
        'description': 'General bacterial indicator of sanitary water quality and environmental contamination.'
    }
}


@st.cache_resource(show_spinner="Loading trained machine learning model...")
def load_trained_model(model_path: str = "water_quality_model.pkl") -> Tuple[Optional[Any], Optional[str]]:
    """
    Loads the saved scikit-learn model using joblib with clean warning suppression.
    Returns (model, error_message).
    """
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found in the workspace."

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Failed to deserialize model '{model_path}': {str(e)}"


def get_model_metadata(model: Any) -> Dict[str, Any]:
    """Extracts verified properties and architecture hyperparameters from the trained model."""
    meta = {
        'model_class': model.__class__.__name__,
        'module': model.__class__.__module__,
        'n_estimators': getattr(model, 'n_estimators', len(getattr(model, 'estimators_', []))),
        'classes': list(getattr(model, 'classes_', [])),
        'n_classes': getattr(model, 'n_classes_', len(getattr(model, 'classes_', []))),
        'n_features_in': getattr(model, 'n_features_in_', None),
        'feature_names_in': list(getattr(model, 'feature_names_in_', [])),
        'has_feature_importances': hasattr(model, 'feature_importances_'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'criterion': getattr(model, 'criterion', 'gini'),
        'class_weight': getattr(model, 'class_weight', None),
        'random_state': getattr(model, 'random_state', None),
        'bootstrap': getattr(model, 'bootstrap', True),
    }
    return meta


def get_feature_importances_df(model: Any) -> Optional[pd.DataFrame]:
    """Returns a pandas DataFrame of feature importances sorted descending."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    features = list(getattr(model, 'feature_names_in_', [f"Feature_{i}" for i in range(len(importances))]))

    df_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Percentage': np.round(importances * 100.0, 2)
    })

    return df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)


def predict_sample(model: Any, input_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Validates input features, constructs the expected input DataFrame,
    and queries the model for prediction and probability estimates.
    """
    expected_feats = list(getattr(model, 'feature_names_in_', []))

    # Match keys flexibly in case of minor unicode differences in conductivity
    aligned_inputs = {}
    for ef in expected_feats:
        if ef in input_dict:
            aligned_inputs[ef] = float(input_dict[ef])
        else:
            # Check by prefix (e.g. CONDUCTIVITY)
            matched = False
            for k, v in input_dict.items():
                if k.split()[0].upper() == ef.split()[0].upper():
                    aligned_inputs[ef] = float(v)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Missing required feature '{ef}' for model prediction.")

    # Create 1-row DataFrame with exact feature names
    X_input = pd.DataFrame([aligned_inputs], columns=expected_feats)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pred_label = model.predict(X_input)[0]

    result = {
        'prediction': pred_label,
        'is_within_range': (pred_label == 'Within pH range'),
        'feature_names': expected_feats,
        'input_values': aligned_inputs
    }

    if hasattr(model, 'predict_proba'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            probas = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
        proba_dict = {cls: float(p) for cls, p in zip(classes, probas)}
        result['probabilities'] = proba_dict
        result['confidence'] = float(np.max(probas))
        result['prob_within'] = proba_dict.get('Within pH range', 0.0)
        result['prob_outside'] = proba_dict.get('Outside pH range', 0.0)

    return result


def evaluate_model_on_labelled_data(model: Any, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, Any]:
    """
    Calculates valid evaluation metrics against the available complete labelled dataset.
    Never invents or hallucinates metrics.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_pred = model.predict(X_eval)

    classes = list(model.classes_)
    acc = float(accuracy_score(y_eval, y_pred))
    cm = confusion_matrix(y_eval, y_pred, labels=classes)
    cr_dict = classification_report(y_eval, y_pred, labels=classes, output_dict=True)

    metrics = {
        'total_samples': len(y_eval),
        'accuracy': acc,
        'accuracy_pct': round(acc * 100.0, 2),
        'classes': classes,
        'confusion_matrix': cm,
        'classification_report': cr_dict,
        'macro_precision': cr_dict['macro avg']['precision'],
        'macro_recall': cr_dict['macro avg']['recall'],
        'macro_f1': cr_dict['macro avg']['f1-score'],
        'weighted_precision': cr_dict['weighted avg']['precision'],
        'weighted_recall': cr_dict['weighted avg']['recall'],
        'weighted_f1': cr_dict['weighted avg']['f1-score'],
        'class_support': {cls: int((y_eval == cls).sum()) for cls in classes},
        'methodology_note': (
            "Metrics are evaluated empirically on all 1,577 complete rows from 'water_dataX.csv' "
            "where ground truth pH is recorded. Since no separate holdout test file was supplied, "
            "this measures full-dataset empirical fidelity. Generalization error on unseen future "
            "basins requires prospective holdout validation."
        )
    }

    return metrics


def check_parameter_thresholds(sample_inputs: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluates individual physicochemical and biological parameters against CPCB and WHO
    surface/drinking water thresholds to provide comprehensive environmental context.
    """
    assessments = []

    for feat_key, val in sample_inputs.items():
        # Match with benchmark keys
        bm = None
        for k in ENVIRONMENTAL_BENCHMARKS:
            if k.split()[0].upper() == feat_key.split()[0].upper():
                bm = ENVIRONMENTAL_BENCHMARKS[k]
                break

        if not bm:
            continue

        status = 'Normal'
        status_color = 'green'
        reason = 'Within typical environmental baseline'

        if 'optimal_min' in bm and val < bm['optimal_min']:
            if val < bm['critical_low']:
                status = 'Critical Low'
                status_color = 'red'
                reason = f"Severely deficient (below {bm['critical_low']} {bm['unit']}). Aquatic hypoxia risk."
            else:
                status = 'Sub-optimal'
                status_color = 'orange'
                reason = f"Lower than healthy surface standard ({bm['standard_min']} {bm['unit']})."
        elif 'optimal_max' in bm and val > bm['optimal_max']:
            if val > bm['critical_high']:
                status = 'Critical High'
                status_color = 'red'
                reason = f"Exceeds critical contamination limit ({bm['critical_high']} {bm['unit']})."
            elif val > bm['standard_max']:
                status = 'Elevated'
                status_color = 'orange'
                reason = f"Above permissible threshold ({bm['standard_max']} {bm['unit']})."
            else:
                status = 'Moderate'
                status_color = 'blue'
                reason = f"Above pristine baseline but within standard limits."
        elif 'optimal' in bm:
            min_opt, max_opt = bm['optimal']
            if val < min_opt or val > max_opt:
                status = 'Thermal Stress'
                status_color = 'orange'
                reason = f"Outside typical natural range ({min_opt}-{max_opt} {bm['unit']})."

        assessments.append({
            'Parameter': bm['name'],
            'Measured Value': f"{val:,.2f} {bm['unit']}",
            'Status': status,
            'Status Color': status_color,
            'Scientific Interpretation': reason,
            'Environmental Context': bm['description']
        })

    return assessments

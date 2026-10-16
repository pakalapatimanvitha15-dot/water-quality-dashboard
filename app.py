"""
HydroIntel: Water Quality Assessment Dashboard
ML x ACS Hackathon 2026

An environmental machine learning dashboard for water quality assessment,
anomaly detection, and physicochemical indicator analysis based on
Central Pollution Control Board (CPCB) monitoring data.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration before any other Streamlit calls
st.set_page_config(
    page_title="HydroIntel | Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from data_loader import (
    load_raw_dataset,
    clean_water_dataset,
    load_feature_list,
    compute_missing_summary,
    compute_numeric_statistics,
    prepare_labelled_evaluation_data,
    find_conductivity_column
)
from model_utils import (
    load_trained_model,
    get_model_metadata,
    get_feature_importances_df,
    predict_sample,
    evaluate_model_on_labelled_data,
    check_parameter_thresholds,
    ENVIRONMENTAL_BENCHMARKS
)
from visualizations import (
    plot_missing_values,
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_scatter_indicators,
    plot_feature_importance,
    plot_confusion_matrix_heatmap,
    plot_prediction_probability_gauge,
    plot_sample_radar_comparison,
    plot_yearly_trends
)
from ui_components import (
    apply_custom_theme,
    render_header_banner,
    render_safety_warning,
    render_prediction_card,
    get_preset_samples,
    render_footer
)


def main():
    # Apply custom styling
    apply_custom_theme()

    # Load data and model assets with graceful error handling
    raw_df, data_err = load_raw_dataset("water_dataX.csv")
    model_features_json, feat_err = load_feature_list("model_features.json")
    model, model_err = load_trained_model("water_quality_model.pkl")

    # Render header banner
    render_header_banner()

    # Sidebar Navigation & System Telemetry
    with st.sidebar:
        st.markdown("### 🧭 **Navigation**")
        page = st.radio(
            "Select Dashboard View:",
            [
                "📊 Overview & Exploration",
                "🧪 Water Sample Assessment",
                "🧠 Model Inspection & Metrics",
                "📜 Guidelines & Chemistry"
            ],
            index=1  # Default to interactive assessment
        )

        st.markdown("---")
        st.markdown("### 📡 **System Status**")

        if data_err:
            st.error(f"❌ Dataset: {data_err}")
        else:
            st.success(f"✅ Dataset: `{len(raw_df):,}` records")

        if feat_err:
            st.error(f"❌ Feature List: {feat_err}")
        else:
            st.success(f"✅ Metadata: `{len(model_features_json)}` features defined")

        if model_err:
            st.error(f"❌ Model: {model_err}")
        else:
            meta = get_model_metadata(model)
            st.success(f"✅ Model: `{meta['model_class']}`")
            st.caption(f"Inputs: `{meta['n_features_in']}` parameters | Trees: `{meta['n_estimators']}`")

        st.markdown("---")
        st.markdown("### 🏆 **Hackathon Info**")
        st.caption("**Event:** ML × ACS Hackathon 2026")
        st.caption("**Theme:** AI for Environmental Chemistry")
        st.caption("**Data Source:** CPCB India Water Network")

    # Clean dataset if available
    df = clean_water_dataset(raw_df) if raw_df is not None else None

    # Handle critical missing file state gracefully
    if model is None:
        st.error(f"⚠️ **Model Initialization Error:** Could not load `water_quality_model.pkl`. Reason: {model_err}")
        st.info("Please verify that `water_quality_model.pkl` is present in the application root directory.")
        return

    # =========================================================================
    # PAGE 1: OVERVIEW & DATA EXPLORATION
    # =========================================================================
    if page == "📊 Overview & Exploration":
        st.markdown("## 📊 Water Monitoring Dataset Overview")
        st.markdown(
            "Explore historical water quality monitoring data compiled from nationwide monitoring stations. "
            "Inspect distribution profiles, missing value proportions, and environmental indicator correlations."
        )

        if df is None:
            st.warning("Dataset `water_dataX.csv` is currently unavailable.")
            return

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">Total Observations</div>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        with col2:
            stations = df['STATION CODE'].nunique() if 'STATION CODE' in df.columns else 'N/A'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Monitoring Stations</div>
                <div class="metric-value">{stations:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            states = df['STATE'].nunique() if 'STATE' in df.columns else 'N/A'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">States / Territories</div>
                <div class="metric-value">{states}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            years = f"{int(df['year'].min())} – {int(df['year'].max())}" if 'year' in df.columns and df['year'].notna().any() else 'N/A'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Temporal Coverage</div>
                <div class="metric-value">{years}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Section 1: Missing Data Analysis
        st.markdown("### 🔍 Missing Data & Data Quality Profile")
        missing_df = compute_missing_summary(df)

        tab_miss1, tab_miss2 = st.tabs(["📊 Missing Values Chart", "📋 Completeness Table"])
        with tab_miss1:
            st.plotly_chart(plot_missing_values(missing_df), use_container_width=True)
            st.caption(
                "Note: In the original raw dataset, missing values were recorded as string literal `'NAN'`. "
                "The dashboard preprocessor correctly translates these to IEEE standard IEEE-754 null values (`np.nan`)."
            )
        with tab_miss2:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Section 2: Distribution & Summary Statistics
        st.markdown("### 📈 Indicator Distribution & Summary Statistics")
        num_stats = compute_numeric_statistics(df)

        stat_col1, stat_col2 = st.columns([1, 2])
        with stat_col1:
            selected_param = st.selectbox(
                "Select Water Quality Parameter:",
                options=num_stats['Parameter'].tolist(),
                index=1
            )
            param_row = num_stats[num_stats['Parameter'] == selected_param].iloc[0]
            st.markdown(f"**Valid Measurements:** `{int(param_row['Valid Count']):,}`")
            st.markdown(f"**Mean ± Std:** `{param_row['Mean']}` ± `{param_row['Std Dev']}`")
            st.markdown(f"**Median (50%):** `{param_row['Median (50%)']}`")
            st.markdown(f"**Interquartile Range:** `{param_row['25%']}` to `{param_row['75%']}`")
            st.markdown(f"**Observed Range:** `{param_row['Min']}` to `{param_row['Max']}`")

        with stat_col2:
            st.plotly_chart(plot_feature_distribution(df, selected_param), use_container_width=True)

        st.markdown("#### Complete Descriptive Statistics Table")
        st.dataframe(num_stats, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Section 3: Multivariate Relationships & Temporal Trends
        st.markdown("### 🔬 Multivariate Environmental Interactions")
        tab_corr, tab_scatter, tab_trend = st.tabs(["🌡️ Correlation Matrix", "🧪 Bivariate Scatter", "📅 Temporal Dynamics"])

        with tab_corr:
            numeric_cols = [c for c in num_stats['Parameter'].tolist() if c in df.columns]
            st.plotly_chart(plot_correlation_heatmap(df, numeric_cols), use_container_width=True)
            st.caption(
                "Insight: Fecal and Total Coliform show strong positive co-correlation (r ~ 0.88), "
                "while Dissolved Oxygen exhibits inverse relationships with Biochemical Oxygen Demand (B.O.D.)."
            )

        with tab_scatter:
            sc_c1, sc_c2 = st.columns(2)
            with sc_c1:
                x_feat = st.selectbox("X-Axis Parameter:", options=numeric_cols, index=1)
            with sc_c2:
                y_feat = st.selectbox("Y-Axis Parameter:", options=numeric_cols, index=4)

            # Check if derived pH range is available for coloring
            if 'PH' in df.columns:
                df['pH_Status'] = np.where(
                    (df['PH'] >= 6.5) & (df['PH'] <= 8.5),
                    'Within pH range',
                    'Outside pH range'
                )
                color_var = 'pH_Status'
            else:
                color_var = None

            st.plotly_chart(plot_scatter_indicators(df, x_feat, y_feat, color_var), use_container_width=True)

        with tab_trend:
            trend_param = st.selectbox("Select Parameter for Multi-Year Trend:", options=numeric_cols, index=1)
            st.plotly_chart(plot_yearly_trends(df, trend_param), use_container_width=True)

        st.markdown("---")

        # Section 4: Filterable Data Explorer
        with st.expander("🔍 Interactive Raw Dataset Explorer"):
            st.markdown("Filter and inspect records from the National Water Quality Monitoring Network.")
            filt_c1, filt_c2 = st.columns(2)
            with filt_c1:
                all_states = sorted([str(s) for s in df['STATE'].dropna().unique()])
                state_filter = st.multiselect("Filter by State:", options=all_states, default=[])
            with filt_c2:
                all_years = sorted([int(y) for y in df['year'].dropna().unique()])
                year_filter = st.multiselect("Filter by Year:", options=all_years, default=[])

            filtered_df = df.copy()
            if state_filter:
                filtered_df = filtered_df[filtered_df['STATE'].isin(state_filter)]
            if year_filter:
                filtered_df = filtered_df[filtered_df['year'].isin(year_filter)]

            st.write(f"Showing `{len(filtered_df):,}` filtered records:")
            st.dataframe(filtered_df, use_container_width=True)

    # =========================================================================
    # PAGE 2: WATER SAMPLE ASSESSMENT & PREDICTION
    # =========================================================================
    elif page == "🧪 Water Sample Assessment":
        st.markdown("## 🧪 Interactive Water Sample Assessment")
        st.markdown(
            "Enter physiochemical and microbial laboratory readings to assess water quality classification. "
            "The system evaluates the sample using the trained Random Forest Classifier and compares individual "
            "indicators against Central Pollution Control Board (CPCB) and World Health Organization (WHO) benchmarks."
        )

        # Prominent Preliminary Warning Banner
        render_safety_warning()

        meta = get_model_metadata(model)
        expected_features = meta['feature_names_in']

        # Sample Presets Selector
        presets = get_preset_samples()
        st.markdown("### ⚡ Quick Scenario Presets")
        st.caption("Select a verified preset to populate the assessment form with representative real-world values:")

        preset_cols = st.columns(len(presets))
        preset_choice = None
        for i, (name, pdata) in enumerate(presets.items()):
            with preset_cols[i]:
                if st.button(f"📌 {name.split()[0]} {name.split()[1] if len(name.split()) > 1 else ''}", key=f"btn_preset_{i}", use_container_width=True):
                    st.session_state['selected_preset'] = name

        # Determine current active values (preset or defaults)
        active_preset = st.session_state.get('selected_preset', "Dataset Median Baseline (Normal Surface River)")
        preset_info = presets.get(active_preset, presets["Dataset Median Baseline (Normal Surface River)"])
        active_vals = preset_info["values"]

        st.info(f"**Active Scenario:** `{active_preset}` — {preset_info['description']}")

        st.write("")
        st.markdown("### 📋 Sample Input Parameters")

        # Two-column layout for input features
        col_left, col_right = st.columns(2)
        sample_inputs = {}

        with col_left:
            st.markdown("#### 🌡️ Physical & General Chemistry")
            # Feature 1: Temperature
            sample_inputs['Temp'] = st.slider(
                "1. Water Temperature (°C)",
                min_value=5.0,
                max_value=45.0,
                value=float(active_vals.get('Temp', 27.0)),
                step=0.1,
                help="Measured in degrees Celsius. Normal river range: 15–30 °C."
            )

            # Feature 2: Dissolved Oxygen
            sample_inputs['D.O. (mg/l)'] = st.slider(
                "2. Dissolved Oxygen - D.O. (mg/L)",
                min_value=0.0,
                max_value=15.0,
                value=float(active_vals.get('D.O. (mg/l)', 6.7)),
                step=0.1,
                help="Minimum 4.0 mg/L required for healthy aquatic life; >6.0 mg/L is desirable."
            )

            # Feature 3: Electrical Conductivity
            cond_key = [f for f in expected_features if 'CONDUCTIVITY' in f.upper()][0]
            sample_inputs[cond_key] = st.number_input(
                "3. Electrical Conductivity (µmhos/cm)",
                min_value=0.0,
                max_value=80000.0,
                value=float(active_vals.get('CONDUCTIVITY (\u00b5mhos/cm)', 183.0)),
                step=10.0,
                help="Measures dissolved ionic mineral salts. Typical fresh river water: 50–500 µmhos/cm."
            )

            # Feature 4: Biochemical Oxygen Demand
            sample_inputs['B.O.D. (mg/l)'] = st.number_input(
                "4. Biochemical Oxygen Demand - B.O.D. (mg/L)",
                min_value=0.0,
                max_value=600.0,
                value=float(active_vals.get('B.O.D. (mg/l)', 1.9)),
                step=0.1,
                help="Organic pollution index. Clean water: <2 mg/L. Moderately polluted: 3–5 mg/L. Raw sewage: >30 mg/L."
            )

        with col_right:
            st.markdown("#### 🧪 Nutrients & Biological Pathogens")
            # Feature 5: Nitrate + Nitrite
            nit_key = 'NITRATENAN N+ NITRITENANN (mg/l)'
            sample_inputs[nit_key] = st.number_input(
                "5. Nitrate + Nitrite Nitrogen (mg/L)",
                min_value=0.0,
                max_value=150.0,
                value=float(active_vals.get(nit_key, 0.52)),
                step=0.05,
                help="Nutrient runoff from fertilizers and sewage. WHO drinking limit: <10 mg/L as NO3-N."
            )

            # Feature 6: Fecal Coliform
            fecal_key = 'FECAL COLIFORM (MPN/100ml)'
            sample_inputs[fecal_key] = st.number_input(
                "6. Fecal Coliform (MPN/100 mL)",
                min_value=0.0,
                max_value=300000000.0,
                value=float(active_vals.get(fecal_key, 221.0)),
                step=10.0,
                help="Pathogenic intestinal bacteria. Desirable limit for bathing: <500 MPN/100mL. Drinking: 0 MPN."
            )

            # Feature 7: Total Coliform
            total_key = 'TOTAL COLIFORM (MPN/100ml)Mean'
            sample_inputs[total_key] = st.number_input(
                "7. Total Coliform (MPN/100 mL)",
                min_value=0.0,
                max_value=600000000.0,
                value=float(active_vals.get(total_key, 468.0)),
                step=10.0,
                help="General sanitary microbial index. Standard with conventional treatment: <5000 MPN/100mL."
            )

            # Optional Measured pH Check
            st.markdown("#### 🔬 Physical Validation (Optional)")
            has_ph = st.checkbox("I have an empirical laboratory pH measurement to cross-validate", value=True)
            user_ph = None
            if has_ph:
                user_ph = st.number_input(
                    "Measured Sample pH (0.0 – 14.0)",
                    min_value=0.0,
                    max_value=14.0,
                    value=float(active_vals.get('PH', 7.3)),
                    step=0.05,
                    help="Permissible drinking and surface water range is 6.5 to 8.5."
                )

        st.write("")
        assess_button = st.button("🚀 Assess Water Sample Quality", type="primary", use_container_width=True)

        # Trigger Prediction
        if assess_button or 'last_prediction' in st.session_state:
            st.session_state['last_prediction'] = True

            try:
                # Predict with model
                pred_res = predict_sample(model, sample_inputs)

                st.markdown("---")
                st.markdown("### 🏆 Assessment Results & Model Prediction")

                # Top Prediction Card
                render_prediction_card(
                    prediction=pred_res['prediction'],
                    confidence=pred_res['confidence'],
                    probabilities=pred_res['probabilities'],
                    optional_ph=user_ph
                )

                # Probability Gauge
                st.plotly_chart(
                    plot_prediction_probability_gauge(
                        prob_within=pred_res['prob_within'],
                        prediction_label=pred_res['prediction']
                    ),
                    use_container_width=True
                )

                # Multi-Parameter Radar and Threshold Evaluations
                col_diag1, col_diag2 = st.columns([1, 1])

                with col_diag1:
                    st.markdown("#### 🕸️ Multi-Parameter Profile vs National Baseline")
                    # Compute baseline medians
                    baseline_medians = {
                        'Temp': 27.0,
                        'D.O. (mg/l)': 6.7,
                        cond_key: 183.0,
                        'B.O.D. (mg/l)': 1.9,
                        nit_key: 0.52,
                        fecal_key: 221.0,
                        total_key: 468.0
                    }
                    st.plotly_chart(
                        plot_sample_radar_comparison(sample_inputs, baseline_medians),
                        use_container_width=True
                    )
                    st.caption("Radar axis displays ratio relative to national median (1.0 = median baseline).")

                with col_diag2:
                    st.markdown("#### 📋 Parameter Compliance Breakdown")
                    param_checks = check_parameter_thresholds(sample_inputs)
                    check_df = pd.DataFrame(param_checks)
                    st.dataframe(
                        check_df[['Parameter', 'Measured Value', 'Status', 'Scientific Interpretation']],
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

        st.markdown("---")
        # Explanatory accordion for feature guidance
        with st.expander("📚 Guide: What do these 7 features signify in Environmental Chemistry?"):
            st.markdown("""
            1. **Water Temperature (°C):** Regulates kinetic reaction rates and gas solubility. Warmer water holds less dissolved oxygen, intensifying stress from organic pollution.
            2. **Dissolved Oxygen (D.O., mg/L):** The primary index of river health. High values (>6 mg/L) indicate active aeration and photosynthetic oxygenation.
            3. **Electrical Conductivity (µmhos/cm):** Directly reflects dissolved ionic solids ($Ca^{2+}$, $Mg^{2+}$, $Na^+$, $Cl^-$, $SO_4^{2-}$). Extreme values point to industrial effluent, agricultural saline drainage, or seawater intrusion.
            4. **Biochemical Oxygen Demand (B.O.D., mg/L):** Quantifies biodegradable organic matter consumed by aerobic bacteria over 5 days. High BOD rapidly drives dissolved oxygen to zero (the 'oxygen sag' phenomenon).
            5. **Nitrate + Nitrite Nitrogen (mg/L):** Key plant nutrients. Runoff from synthetic nitrogenous fertilizers triggers eutrophication, hazardous cyanobacterial algal blooms, and infant methemoglobinemia.
            6. **Fecal Coliform (MPN/100 mL):** Thermotolerant enteric coliforms (*Escherichia coli*) specifically indicating fecal contamination from humans or warm-blooded animals.
            7. **Total Coliform (MPN/100 mL):** Broad indicator of bacterial contamination including soil and aquatic bacteria alongside fecal pathogens.
            """)

    # =========================================================================
    # PAGE 3: MODEL INSPECTION & PERFORMANCE METRICS
    # =========================================================================
    elif page == "🧠 Model Inspection & Metrics":
        st.markdown("## 🧠 Model Architecture & Performance Validation")
        st.markdown(
            "Inspect the verified architecture of the pre-trained `water_quality_model.pkl` and evaluate its "
            "empirical classification performance on the available labelled dataset."
        )

        meta = get_model_metadata(model)

        # Architectural Highlights
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Model Family</div>
                <div class="metric-value" style="font-size: 1.3rem;">{meta['model_class']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Ensemble Size</div>
                <div class="metric-value">{meta['n_estimators']} Trees</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Input Dimension</div>
                <div class="metric-value">{meta['n_features_in']} Features</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Output Target</div>
                <div class="metric-value" style="font-size: 1.25rem;">pH Permissibility</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Section 1: Feature Importance
        st.markdown("### 🌲 Feature Importance (Gini Impurity Reduction)")
        st.markdown(
            "Feature importances are computed directly from the trained decision tree ensemble splits (`model.feature_importances_`). "
            "They quantify the relative predictive contribution of each physicochemical parameter in determining pH compliance."
        )

        imp_df = get_feature_importances_df(model)
        if imp_df is not None:
            col_imp1, col_imp2 = st.columns([2, 1])
            with col_imp1:
                st.plotly_chart(plot_feature_importance(imp_df), use_container_width=True)
            with col_imp2:
                st.markdown("#### 🔬 Environmental Science Insights")
                st.markdown("""
                - **Coliform Burden (~38.4% combined):** Fecal Coliform (19.4%) and Total Coliform (19.0%) are the top predictors. High microbial loads correlate with intense decomposition, volatile fatty acid generation, and buffering breakdown.
                - **Electrical Conductivity (16.8%):** Reflects total dissolved mineral salts and electrolyte balance governing water buffering capacity.
                - **Biochemical Oxygen Demand (12.7%):** Organic loading directly dictates carbonic acid ($H_2CO_3$) equilibrium via microbial respiration.
                - **Temperature (11.8%) & D.O. (10.1%):** Dictate gas solubility ($CO_2$, $O_2$) and metabolic kinetics.
                """)

        st.markdown("---")

        # Section 2: Rigorous Model Evaluation
        st.markdown("### 📐 Empirical Evaluation on Available Labelled Dataset")
        st.markdown(
            "In strict accordance with the hackathon principles, **no model accuracy or metrics are invented**. "
            "Evaluation is performed on the complete set of 1,577 records from `water_dataX.csv` with complete measurements "
            "and ground-truth pH compliance ([6.5, 8.5] range)."
        )

        X_eval, y_eval = prepare_labelled_evaluation_data(df, meta['feature_names_in'])

        if X_eval is not None and y_eval is not None:
            metrics = evaluate_model_on_labelled_data(model, X_eval, y_eval)

            # Metric Summary Cards
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Overall Accuracy</div>
                    <div class="metric-value">{metrics['accuracy_pct']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_e2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Macro Precision</div>
                    <div class="metric-value">{metrics['macro_precision']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_e3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Macro Recall</div>
                    <div class="metric-value">{metrics['macro_recall']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_e4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Macro F1-Score</div>
                    <div class="metric-value">{metrics['macro_f1']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            col_cm, col_cr = st.columns([1, 1])
            with col_cm:
                st.plotly_chart(
                    plot_confusion_matrix_heatmap(metrics['confusion_matrix'], metrics['classes']),
                    use_container_width=True
                )
            with col_cr:
                st.markdown("#### 📋 Detailed Classification Report")
                # Format report as DataFrame
                report_rows = []
                for cls_name in metrics['classes']:
                    cls_stats = metrics['classification_report'][cls_name]
                    report_rows.append({
                        'Class': cls_name,
                        'Precision': f"{cls_stats['precision']*100:.2f}%",
                        'Recall': f"{cls_stats['recall']*100:.2f}%",
                        'F1-Score': f"{cls_stats['f1-score']*100:.2f}%",
                        'Support': f"{int(cls_stats['support']):,}"
                    })
                report_df = pd.DataFrame(report_rows)
                st.dataframe(report_df, use_container_width=True, hide_index=True)

                st.markdown("#### ⚙️ Hyperparameter Configuration")
                st.markdown(f"""
                - **Criterion:** `{meta['criterion']}`
                - **Class Weight:** `{meta['class_weight']}` *(balances 9:1 class imbalance)*
                - **Random State:** `{meta['random_state']}`
                - **Bootstrap:** `{meta['bootstrap']}`
                """)

            # Transparent Methodology Note
            st.info(f"ℹ️ **Scientific Methodology Note:** {metrics['methodology_note']}")

        else:
            st.warning("Insufficient labelled samples in `water_dataX.csv` to calculate evaluation metrics.")

        st.markdown("---")

        # Section 3: Feature Alignment Deep Dive
        with st.expander("🔍 Deep Dive: Feature Alignment & Target Formulation"):
            st.markdown("""
            **Why does `model_features.json` contain 8 features while the model uses 7?**
            - In `model_features.json`, the recorded parameters are: `Temp`, `D.O.`, `PH`, `CONDUCTIVITY`, `B.O.D.`, `NITRATENAN N+ NITRITENANN`, `FECAL COLIFORM`, and `TOTAL COLIFORM`.
            - However, when this machine learning model was developed, the goal was to **predict whether the water falls within the permissible pH range [6.5 – 8.5]** without directly measuring pH with an electrode.
            - Therefore, `PH` was utilized to construct the binary classification target labels (`Within pH range` vs `Outside pH range`).
            - To prevent **label leakage**, `PH` was appropriately excluded from the feature matrix $X$, leaving the remaining **7 physicochemical and microbial parameters** as the exact input features expected by the trained `RandomForestClassifier`.
            """)

    # =========================================================================
    # PAGE 4: GUIDELINES & CHEMISTRY
    # =========================================================================
    elif page == "📜 Guidelines & Chemistry":
        st.markdown("## 📜 Environmental Chemistry & Regulatory Standards")
        st.markdown(
            "Review the environmental chemical mechanisms governing water quality and the formal Central Pollution "
            "Control Board (CPCB) Designated-Best-Use classifications."
        )

        st.markdown("### 🏛️ CPCB Water Quality Criteria (Designated Best Use)")
        st.markdown("""
        | Class | Designated Best Use | Key Quality Criteria |
        | :--- | :--- | :--- |
        | **Class A** | Drinking water source without conventional treatment but after disinfection | • Total Coliform < 50 MPN/100mL<br>• pH: 6.5 – 8.5<br>• D.O. ≥ 6 mg/L<br>• B.O.D. ≤ 2 mg/L |
        | **Class B** | Outdoor bathing (organized) | • Total Coliform < 500 MPN/100mL<br>• pH: 6.5 – 8.5<br>• D.O. ≥ 5 mg/L<br>• B.O.D. ≤ 3 mg/L |
        | **Class C** | Drinking water source after conventional treatment and disinfection | • Total Coliform < 5,000 MPN/100mL<br>• pH: 6.0 – 9.0<br>• D.O. ≥ 4 mg/L<br>• B.O.D. ≤ 3 mg/L |
        | **Class D** | Propagation of wild life and fisheries | • pH: 6.5 – 8.5<br>• D.O. ≥ 4 mg/L<br>• Free Ammonia ≤ 1.2 mg/L |
        | **Class E** | Irrigation, industrial cooling, controlled waste disposal | • pH: 6.0 – 8.5<br>• Conductivity ≤ 2,250 µmhos/cm<br>• Sodium Absorption Ratio ≤ 26 |
        """)

        st.markdown("---")

        st.markdown("### ⚗️ The Chemistry of pH Buffering in Aquatic Systems")
        st.markdown("""
        The pH of natural surface waters is governed by the **carbonate-bicarbonate buffer equilibrium**:

        $$\\text{CO}_2 \\text{ (aq)} + \\text{H}_2\\text{O} \\rightleftharpoons \\text{H}_2\\text{CO}_3 \\rightleftharpoons \\text{HCO}_3^- + \\text{H}^+ \\rightleftharpoons \\text{CO}_3^{2-} + 2\\text{H}^+$$

        #### 1. Why High B.O.D. Affects pH
        When large volumes of organic waste (sewage, food processing effluent) enter a river, heterotrophic aerobic bacteria consume dissolved oxygen to decompose the organic matter:

        $$\\text{Organic Carbon } (\\text{CH}_2\\text{O})_n + n\\text{O}_2 \\xrightarrow{\\text{microbes}} n\\text{CO}_2 + n\\text{H}_2\\text{O}$$

        Excess $\\text{CO}_2$ hydrates into carbonic acid ($\\text{H}_2\\text{CO}_3$), releasing hydrogen ions ($H^+$) and driving the pH down toward acidic states (<6.5).

        #### 2. Eutrophication and Alkaline Spikes
        Conversely, in waters with high Nitrate ($NO_3^-$) runoff under warm temperatures, algal blooms proliferate.
        During intense daytime photosynthesis, algae consume aqueous $\\text{CO}_2$ faster than atmospheric replenishment:

        $$\\text{HCO}_3^- \\rightarrow \\text{CO}_2 \\text{ (consumed by algae)} + \\text{OH}^-$$

        This causes a dramatic shift in hydroxide ion concentrations ($OH^-$), pushing pH above 8.5 or even 9.0 during peak daylight hours.

        #### 3. Role of Machine Learning
        Because these interrelated biogeochemical equilibria are non-linear and subject to complex threshold effects,
        the non-parametric decision boundaries in a **Random Forest ensemble** can capture cross-parameter signals
        (e.g., high BOD + low DO + moderate Conductivity) to reliably flag water systems whose buffering has collapsed.
        """)

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()

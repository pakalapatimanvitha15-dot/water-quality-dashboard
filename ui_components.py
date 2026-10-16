"""
UI Components, Custom Styling, and Alert Banners for the Water Quality Dashboard.
"""

from typing import Dict, List, Optional, Any
import streamlit as st


def apply_custom_theme() -> None:
    """Injects high-polish environmental science CSS with dark navy and teal accents."""
    custom_css = """
    <style>
        /* Base page variables and typography */
        :root {
            --navy-deep: #0a1128;
            --navy-sidebar: #0d1b2a;
            --navy-card: #1b263b;
            --navy-border: #415a77;
            --teal-bright: #00f5d4;
            --teal-cyan: #00b4d8;
            --teal-deep: #0077b6;
            --text-light: #e0e1dd;
            --green-safe: #06d6a0;
            --coral-alert: #ef476f;
            --amber-warn: #ffd166;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0b1726 !important;
            border-right: 1px solid rgba(0, 180, 216, 0.25) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #00f5d4 !important;
        }

        /* Main content container */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Top Header Card */
        .header-card {
            background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.92));
            border: 1px solid rgba(0, 245, 212, 0.35);
            box-shadow: 0 8px 32px 0 rgba(0, 180, 216, 0.12);
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 24px;
        }
        .header-title {
            color: #00f5d4;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-subtitle {
            color: #90e0ef;
            font-size: 1.05rem;
            margin-top: 6px;
            margin-bottom: 14px;
        }

        /* Badges */
        .badge-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
        }
        .hackathon-badge {
            background: rgba(0, 119, 182, 0.3);
            border: 1px solid rgba(0, 245, 212, 0.4);
            color: #00f5d4;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        /* Custom Cards */
        .glass-card {
            background: rgba(27, 38, 59, 0.65);
            border: 1px solid rgba(0, 180, 216, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 18px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(0, 245, 212, 0.4);
            box-shadow: 0 4px 20px rgba(0, 180, 216, 0.15);
        }

        /* Metric Box */
        .metric-box {
            background: rgba(13, 27, 42, 0.7);
            border: 1px solid rgba(0, 180, 216, 0.25);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00f5d4;
            margin: 4px 0;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #90e0ef;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Safety Warning Banner */
        .disclaimer-banner {
            background: rgba(239, 71, 111, 0.12);
            border: 1px solid rgba(239, 71, 111, 0.45);
            border-left: 5px solid #ef476f;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 16px 0 24px 0;
            color: #f8f9fa;
        }
        .disclaimer-title {
            color: #ef476f;
            font-weight: 700;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }
        .disclaimer-text {
            font-size: 0.86rem;
            line-height: 1.45;
            color: #e0e1dd;
            margin: 0;
        }

        /* Prediction Result Cards */
        .pred-card-safe {
            background: rgba(6, 214, 160, 0.12);
            border: 1.5px solid #06d6a0;
            border-radius: 12px;
            padding: 22px;
            margin: 16px 0;
        }
        .pred-card-outside {
            background: rgba(239, 71, 111, 0.12);
            border: 1.5px solid #ef476f;
            border-radius: 12px;
            padding: 22px;
            margin: 16px 0;
        }
        .pred-title-safe {
            color: #06d6a0;
            font-size: 1.4rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .pred-title-outside {
            color: #ef476f;
            font-size: 1.4rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Scientific note callout */
        .science-note {
            background: rgba(0, 119, 182, 0.15);
            border-left: 4px solid #00b4d8;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.88rem;
            color: #caf0f8;
            margin-top: 10px;
        }

        /* Footer */
        .dashboard-footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(0, 180, 216, 0.2);
            text-align: center;
            color: #778da9;
            font-size: 0.85rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header_banner() -> None:
    """Renders the top branding header with Hackathon badges."""
    st.markdown("""
    <div class="header-card">
        <div class="header-title">
            🌊 HydroIntel: Water Quality Assessment Dashboard
        </div>
        <div class="header-subtitle">
            Environmental Machine Learning Screening System • Central Pollution Control Board (CPCB) Data
        </div>
        <div class="badge-container">
            <span class="hackathon-badge">🏆 ML × ACS Hackathon 2026</span>
            <span class="hackathon-badge">🌲 Random Forest Classifier (100 Estimators)</span>
            <span class="hackathon-badge">🧪 7 Core Physiochemical & Biological Features</span>
            <span class="hackathon-badge">🎯 Target: Standard pH Permissible Range [6.5 – 8.5]</span>
            <span class="hackathon-badge">🛡️ Rigorous Scientific Validation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_safety_warning(compact: bool = False) -> None:
    """Renders mandatory preliminary screening disclaimer."""
    if compact:
        st.markdown("""
        <div class="disclaimer-banner" style="padding: 10px 14px; margin: 10px 0;">
            <div class="disclaimer-title" style="font-size: 0.88rem;">
                ⚠️ PRELIMINARY SCREENING NOTICE
            </div>
            <p class="disclaimer-text" style="font-size: 0.82rem;">
                ML prediction is an algorithmic screening tool and <b>not a drinking-water safety certificate</b>.
                Certified laboratory analytical testing (ISO/BIS/EPA) is required before human consumption.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="disclaimer-banner">
            <div class="disclaimer-title">
                ⚠️ CRITICAL COMPLIANCE NOTICE: PRELIMINARY SCREENING ONLY
            </div>
            <p class="disclaimer-text">
                This machine learning model provides <b>preliminary environmental screening and anomaly detection</b>
                developed for the ML × ACS Hackathon 2026. It assesses whether water parameters correlate with acceptable
                surface/potable pH ranges [6.5 – 8.5].
                <br><br>
                <b>This application DOES NOT issue drinking-water safety certificates, potability approvals, or regulatory compliance sign-offs.</b>
                Water safety certification requires comprehensive laboratory testing for heavy metals, pesticides, emerging contaminants,
                and microbial cultures by an accredited laboratory under BIS (IS 10500:2012) or WHO drinking water standards.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_card(
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    optional_ph: Optional[float] = None
) -> None:
    """Displays the verified prediction output with environmental interpretation."""
    is_safe = (prediction == 'Within pH range')
    card_class = "pred-card-safe" if is_safe else "pred-card-outside"
    title_class = "pred-title-safe" if is_safe else "pred-title-outside"
    icon = "✅" if is_safe else "⚠️"
    headline = "WITHIN PERMISSIBLE pH RANGE [6.5 – 8.5]" if is_safe else "OUTSIDE ACCEPTABLE pH RANGE (<6.5 or >8.5)"

    prob_within = probabilities.get('Within pH range', 0.0) * 100.0
    prob_outside = probabilities.get('Outside pH range', 0.0) * 100.0

    st.markdown(f"""
    <div class="{card_class}">
        <div class="{title_class}">
            {icon} Actual Model Output: {prediction.upper()}
        </div>
        <p style="font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 8px;">
            Status: {headline}
        </p>
        <p style="font-size: 0.92rem; color: #e0e1dd; margin-bottom: 12px;">
            <b>Confidence Score:</b> {confidence * 100:.1f}% &nbsp;|&nbsp; 
            <b>Within Range Probability:</b> {prob_within:.1f}% &nbsp;|&nbsp; 
            <b>Outside Range Probability:</b> {prob_outside:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Scientific rationale
    if is_safe:
        st.markdown("""
        <div class="science-note">
            <b>🔬 Chemical & Ecological Interpretation:</b>
            The sample's input indicators (Conductivity, BOD, Coliforms, Dissolved Oxygen) are chemically consistent
            with stable carbonate-bicarbonate buffering capacity in surface waters (pH 6.5 – 8.5).
            Aerobic self-purification is likely functioning within standard ecological equilibrium.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="science-note" style="border-left-color: #ef476f; background: rgba(239, 71, 111, 0.15);">
            <b>🔬 Chemical & Ecological Interpretation:</b>
            The multi-parameter profile indicates chemical stress or buffering destabilization.
            Potential triggers include organic over-enrichment (elevated BOD driving anaerobic fermentation and carbonic acid),
            acid rock drainage, agricultural nitrate runoff, or industrial caustic/acidic discharge.
        </div>
        """, unsafe_allow_html=True)

    # Comparison if user provided optional measured pH
    if optional_ph is not None:
        ph_in_range = (6.5 <= optional_ph <= 8.5)
        st.write("")
        st.markdown(f"**Laboratory Measured pH Check:** Measured Value = `{optional_ph:.2f}`")
        if (is_safe and ph_in_range) or (not is_safe and not ph_in_range):
            st.success(f"🎯 **Model Agreement:** The ML model prediction directly corroborates the physical measured pH ({optional_ph:.2f})!")
        else:
            st.warning(f"⚖️ **Divergence Observed:** Physical pH is `{optional_ph:.2f}` ({'Within' if ph_in_range else 'Outside'} range), while ML prediction was `{prediction}`. This indicates potential pre-symptomatic buffering changes or complex ionic mixtures.")


def get_preset_samples() -> Dict[str, Dict[str, Any]]:
    """Returns realistic water sample presets with verified real-world ranges."""
    return {
        "Dataset Median Baseline (Normal Surface River)": {
            "description": "Represents the 50th percentile median baseline across 1,991 Indian monitoring stations.",
            "values": {
                'Temp': 27.0,
                'D.O. (mg/l)': 6.7,
                'CONDUCTIVITY (\u00b5mhos/cm)': 183.0,
                'B.O.D. (mg/l)': 1.9,
                'NITRATENAN N+ NITRITENANN (mg/l)': 0.52,
                'FECAL COLIFORM (MPN/100ml)': 221.0,
                'TOTAL COLIFORM (MPN/100ml)Mean': 468.0,
                'PH': 7.3
            }
        },
        "Pristine Headwater Stream": {
            "description": "Clean, highly oxygenated mountain tributary with minimal anthropogenic impact.",
            "values": {
                'Temp': 18.5,
                'D.O. (mg/l)': 8.4,
                'CONDUCTIVITY (\u00b5mhos/cm)': 85.0,
                'B.O.D. (mg/l)': 1.1,
                'NITRATENAN N+ NITRITENANN (mg/l)': 0.20,
                'FECAL COLIFORM (MPN/100ml)': 12.0,
                'TOTAL COLIFORM (MPN/100ml)Mean': 35.0,
                'PH': 7.4
            }
        },
        "Agricultural Runoff Stream": {
            "description": "Watershed receiving intensive synthetic fertilizer wash and moderate livestock effluent.",
            "values": {
                'Temp': 28.5,
                'D.O. (mg/l)': 5.2,
                'CONDUCTIVITY (\u00b5mhos/cm)': 650.0,
                'B.O.D. (mg/l)': 4.2,
                'NITRATENAN N+ NITRITENANN (mg/l)': 8.50,
                'FECAL COLIFORM (MPN/100ml)': 1800.0,
                'TOTAL COLIFORM (MPN/100ml)Mean': 5200.0,
                'PH': 7.8
            }
        },
        "Severe Sewage / Organic Contamination": {
            "description": "Severely polluted urban canal with untreated domestic sewage and septic conditions.",
            "values": {
                'Temp': 31.0,
                'D.O. (mg/l)': 1.2,
                'CONDUCTIVITY (\u00b5mhos/cm)': 1450.0,
                'B.O.D. (mg/l)': 38.0,
                'NITRATENAN N+ NITRITENANN (mg/l)': 14.50,
                'FECAL COLIFORM (MPN/100ml)': 180000.0,
                'TOTAL COLIFORM (MPN/100ml)Mean': 450000.0,
                'PH': 6.3
            }
        },
        "Dataset Anomaly Station (Confirmed Outside Range)": {
            "description": "Historical monitoring record from River Assonora (Goa) exhibiting an out-of-range acidic event (pH 6.4).",
            "values": {
                'Temp': 27.9,
                'D.O. (mg/l)': 6.7,
                'CONDUCTIVITY (\u00b5mhos/cm)': 93.0,
                'B.O.D. (mg/l)': 1.4,
                'NITRATENAN N+ NITRITENANN (mg/l)': 0.10,
                'FECAL COLIFORM (MPN/100ml)': 2147.0,
                'TOTAL COLIFORM (MPN/100ml)Mean': 3433.0,
                'PH': 6.4
            }
        }
    }


def render_footer() -> None:
    """Renders the standard Hackathon footer."""
    st.markdown("""
    <div class="dashboard-footer">
        <b>HydroIntel Water Quality Assessment Dashboard</b> • Developed for the ML × ACS Hackathon 2026<br>
        Built with Streamlit, scikit-learn, pandas & Plotly • Grounded in Central Pollution Control Board (CPCB) Research
    </div>
    """, unsafe_allow_html=True)

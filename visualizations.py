"""
Plotly visualization functions for the Water Quality Assessment Dashboard.
Uses a cohesive environmental science theme with navy, teal, and cyan palettes.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color Theme Constants
COLOR_NAVY_DARK = "#0d1b2a"
COLOR_NAVY_CARD = "#1b263b"
COLOR_TEAL_DARK = "#0077b6"
COLOR_TEAL_PRIMARY = "#00b4d8"
COLOR_CYAN_ACCENT = "#00f5d4"
COLOR_GREEN_GOOD = "#06d6a0"
COLOR_AMBER_WARN = "#ffd166"
COLOR_CORAL_ALERT = "#ef476f"
COLOR_TEXT_LIGHT = "#e0e1dd"
COLOR_GRID = "rgba(224, 225, 221, 0.15)"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color=COLOR_TEXT_LIGHT, size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    legend=dict(
        bgcolor="rgba(27, 38, 59, 0.7)",
        bordercolor=COLOR_GRID,
        borderwidth=1,
        font=dict(color=COLOR_TEXT_LIGHT)
    )
)


def apply_theme(fig: go.Figure, **kwargs) -> go.Figure:
    """Applies standard dashboard theme and merges custom layout properties."""
    fig.update_layout(**BASE_LAYOUT)
    if kwargs:
        fig.update_layout(**kwargs)
    return fig


def plot_missing_values(summary_df: pd.DataFrame) -> go.Figure:
    """Plots a modern horizontal bar chart of missing value percentages."""
    df_sorted = summary_df.sort_values(by='Missing (%)', ascending=True)

    colors = [
        COLOR_CORAL_ALERT if pct > 30 else (COLOR_AMBER_WARN if pct > 10 else COLOR_TEAL_PRIMARY)
        for pct in df_sorted['Missing (%)']
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted['Missing (%)'],
        y=df_sorted['Column'],
        orientation='h',
        marker=dict(color=colors, line=dict(color=COLOR_CYAN_ACCENT, width=0.5)),
        text=[f"{pct:.1f}% ({cnt:,} rows)" for pct, cnt in zip(df_sorted['Missing (%)'], df_sorted['Missing Count'])],
        textposition='outside',
        textfont=dict(size=11, color=COLOR_TEXT_LIGHT),
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>"
    ))

    apply_theme(
        fig,
        title=dict(text="<b>Data Completeness & Missing Value Profile</b>", font=dict(size=15, color=COLOR_CYAN_ACCENT)),
        xaxis_title="Missing Percentage (%)",
        yaxis_title="",
        height=400
    )
    fig.update_xaxes(range=[0, max(float(summary_df['Missing (%)'].max()) * 1.25, 45.0)], gridcolor=COLOR_GRID)
    return fig


def plot_feature_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Creates a distribution histogram with an integrated box plot for a numeric parameter."""
    series = df[column].dropna()
    if len(series) == 0:
        return go.Figure()

    q99 = series.quantile(0.99)
    use_log = (q99 > 5000 and series.max() > 20000)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=series,
        nbinsx=40,
        marker=dict(
            color=COLOR_TEAL_PRIMARY,
            line=dict(color=COLOR_CYAN_ACCENT, width=0.8)
        ),
        opacity=0.85,
        name="Frequency"
    ))

    median_val = float(series.median())
    mean_val = float(series.mean())

    fig.add_vline(
        x=median_val, 
        line_dash="dash", 
        line_color=COLOR_CYAN_ACCENT, 
        annotation_text=f"Median: {median_val:,.2f}",
        annotation_position="top right"
    )

    if not use_log and mean_val < series.quantile(0.95):
        fig.add_vline(
            x=mean_val, 
            line_dash="dot", 
            line_color=COLOR_AMBER_WARN, 
            annotation_text=f"Mean: {mean_val:,.2f}",
            annotation_position="top left"
        )

    apply_theme(
        fig,
        title=dict(text=f"<b>Distribution of {column}</b>", font=dict(size=14, color=COLOR_CYAN_ACCENT)),
        xaxis_title=column,
        yaxis_title="Observation Count",
        height=360,
        bargap=0.1
    )

    if use_log:
        fig.update_xaxes(type="log", title=f"{column} (Log Scale)")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Creates an annotated correlation matrix heatmap for the key indicators."""
    available_feats = [f for f in features if f in df.columns]
    corr_df = df[available_feats].corr().round(2)

    short_labels = [
        f.replace(' (\u00b5mhos/cm)', '').replace(' (mg/l)', '').replace(' (MPN/100ml)', '').replace('NITRATENAN N+ NITRITENANN', 'NO3+NO2')
        for f in available_feats
    ]

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=short_labels,
        y=short_labels,
        colorscale=[
            [0.0, COLOR_NAVY_DARK],
            [0.5, COLOR_NAVY_CARD],
            [0.75, COLOR_TEAL_DARK],
            [1.0, COLOR_CYAN_ACCENT]
        ],
        text=corr_df.values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#ffffff"),
        colorbar=dict(title=dict(text="r", font=dict(color=COLOR_TEXT_LIGHT))),
        hoverongaps=False
    ))

    apply_theme(
        fig,
        title=dict(text="<b>Pairwise Indicator Correlation Matrix (Pearson)</b>", font=dict(size=15, color=COLOR_CYAN_ACCENT)),
        height=450,
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig


def plot_scatter_indicators(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    color_col: Optional[str] = None
) -> go.Figure:
    """Plots interactive bivariate scatter diagram between physicochemical indicators."""
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    x_cap = plot_df[x_col].quantile(0.99)
    y_cap = plot_df[y_col].quantile(0.99)
    plot_df = plot_df[(plot_df[x_col] <= x_cap) & (plot_df[y_col] <= y_cap)]

    if color_col and color_col in plot_df.columns:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map={
                'Within pH range': COLOR_GREEN_GOOD,
                'Outside pH range': COLOR_CORAL_ALERT
            },
            hover_name='LOCATIONS',
            hover_data=['STATE', 'year']
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            hover_name='LOCATIONS',
            hover_data=['STATE', 'year']
        )
        fig.update_traces(marker=dict(color=COLOR_TEAL_PRIMARY, size=7, opacity=0.7, line=dict(color=COLOR_CYAN_ACCENT, width=0.5)))

    apply_theme(
        fig,
        title=dict(text=f"<b>Bivariate Relationship: {x_col} vs {y_col}</b>", font=dict(size=14, color=COLOR_CYAN_ACCENT)),
        height=400
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Plots a styled horizontal bar chart of Random Forest feature importances."""
    df_sorted = importance_df.sort_values(by='Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted['Importance'],
        y=df_sorted['Feature'],
        orientation='h',
        marker=dict(
            color=df_sorted['Importance'],
            colorscale=[[0, COLOR_NAVY_CARD], [0.5, COLOR_TEAL_DARK], [1, COLOR_CYAN_ACCENT]],
            line=dict(color=COLOR_CYAN_ACCENT, width=1)
        ),
        text=[f"{pct:.1f}%" for pct in df_sorted['Percentage']],
        textposition='outside',
        textfont=dict(size=11, color=COLOR_TEXT_LIGHT),
        hovertemplate="<b>%{y}</b><br>Gini Importance: %{x:.4f} (%{text})<extra></extra>"
    ))

    apply_theme(
        fig,
        title=dict(text="<b>Model Feature Importance (Mean Decrease in Impurity)</b>", font=dict(size=15, color=COLOR_CYAN_ACCENT)),
        xaxis_title="Relative Gini Importance",
        yaxis_title="",
        height=420
    )
    fig.update_xaxes(range=[0, max(float(importance_df['Importance'].max()) * 1.25, 0.25)], gridcolor=COLOR_GRID)
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, classes: List[str]) -> go.Figure:
    """Creates a stylized confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted {c}" for c in classes],
        y=[f"Actual {c}" for c in classes],
        colorscale=[
            [0.0, COLOR_NAVY_DARK],
            [0.3, COLOR_NAVY_CARD],
            [0.7, COLOR_TEAL_DARK],
            [1.0, COLOR_CYAN_ACCENT]
        ],
        text=cm,
        texttemplate="<b>%{text:,}</b>",
        textfont=dict(size=16, color="#ffffff"),
        hoverongaps=False
    ))

    apply_theme(
        fig,
        title=dict(text="<b>Empirical Confusion Matrix on Available Labelled Dataset</b>", font=dict(size=14, color=COLOR_CYAN_ACCENT)),
        height=380,
        margin=dict(l=80, r=40, t=50, b=50)
    )
    return fig


def plot_prediction_probability_gauge(prob_within: float, prediction_label: str) -> go.Figure:
    """Creates a dual horizontal probability breakdown bar chart for the prediction."""
    prob_outside = 1.0 - prob_within

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=['Model Output Probability'],
        x=[prob_within * 100],
        name='Within pH Range [6.5 - 8.5]',
        orientation='h',
        marker=dict(color=COLOR_GREEN_GOOD),
        text=[f"Within pH Range: {prob_within*100:.1f}%"],
        textposition='inside',
        textfont=dict(color='#ffffff', size=13, family='Arial Black')
    ))

    fig.add_trace(go.Bar(
        y=['Model Output Probability'],
        x=[prob_outside * 100],
        name='Outside pH Range (<6.5 or >8.5)',
        orientation='h',
        marker=dict(color=COLOR_CORAL_ALERT),
        text=[f"Outside pH Range: {prob_outside*100:.1f}%"] if prob_outside > 0.08 else [""],
        textposition='inside',
        textfont=dict(color='#ffffff', size=12, family='Arial Black')
    ))

    apply_theme(
        fig,
        barmode='stack',
        title=dict(
            text=f"<b>Predicted State Probability Distribution: {prediction_label}</b>",
            font=dict(size=14, color=COLOR_CYAN_ACCENT)
        ),
        yaxis=dict(showticklabels=False),
        height=180,
        margin=dict(l=20, r=20, t=45, b=25),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.5, xanchor='center', x=0.5)
    )
    fig.update_xaxes(range=[0, 100], title="Probability Confidence (%)", gridcolor=COLOR_GRID)
    return fig


def plot_sample_radar_comparison(
    sample_inputs: Dict[str, float],
    baseline_medians: Dict[str, float]
) -> go.Figure:
    """
    Renders a normalized spider / radar chart comparing the sample against national baseline medians.
    Values are scaled relative to baseline (1.0 = median).
    """
    categories = []
    sample_ratios = []
    baseline_ratios = []

    for key, val in sample_inputs.items():
        # Match with baseline keys
        matched_k = None
        for bk in baseline_medians:
            if bk.split()[0].upper() == key.split()[0].upper():
                matched_k = bk
                break

        if matched_k and baseline_medians[matched_k] > 0:
            short_name = key.split()[0]
            if 'CONDUCTIVITY' in key.upper():
                short_name = 'Cond'
            elif 'NITRATE' in key.upper():
                short_name = 'Nitrate'
            elif 'FECAL' in key.upper():
                short_name = 'Fecal'
            elif 'TOTAL' in key.upper():
                short_name = 'Total'

            categories.append(short_name)
            ratio = val / baseline_medians[matched_k]
            capped_val = min(max(ratio, 0.05), 10.0)
            sample_ratios.append(capped_val)
            baseline_ratios.append(1.0)

    if not categories:
        return go.Figure()

    categories_closed = categories + [categories[0]]
    sample_closed = sample_ratios + [sample_ratios[0]]
    baseline_closed = baseline_ratios + [baseline_ratios[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 180, 216, 0.15)',
        line=dict(color=COLOR_TEAL_PRIMARY, dash='dot', width=1.5),
        name='National Median Baseline (1.0x)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=sample_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 245, 212, 0.25)',
        line=dict(color=COLOR_CYAN_ACCENT, width=2.5),
        name='Assessed Sample Ratio'
    ))

    apply_theme(
        fig,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(sample_ratios + [2.0]) * 1.15],
                gridcolor=COLOR_GRID,
                linecolor=COLOR_GRID
            ),
            angularaxis=dict(
                gridcolor=COLOR_GRID,
                linecolor=COLOR_GRID
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=dict(text="<b>Multi-Parameter Profile vs. National River Baseline</b>", font=dict(size=14, color=COLOR_CYAN_ACCENT)),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_yearly_trends(df: pd.DataFrame, parameter: str) -> go.Figure:
    """Plots annual median trend and interquartile envelope over time (2003-2014)."""
    if 'year' not in df.columns or parameter not in df.columns:
        return go.Figure()

    valid_df = df.dropna(subset=['year', parameter]).copy()
    valid_df['year'] = valid_df['year'].astype(int)

    yearly_stats = valid_df.groupby('year')[parameter].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count='count'
    ).reset_index()

    yearly_stats = yearly_stats.sort_values(by='year')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['q75'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['q25'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 180, 216, 0.15)',
        name='25th - 75th Percentile Range'
    ))

    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['median'],
        mode='lines+markers',
        line=dict(color=COLOR_CYAN_ACCENT, width=3),
        marker=dict(size=7, color=COLOR_CYAN_ACCENT),
        name='Annual National Median'
    ))

    apply_theme(
        fig,
        title=dict(text=f"<b>Temporal Dynamics of {parameter} (2003 – 2014)</b>", font=dict(size=14, color=COLOR_CYAN_ACCENT)),
        xaxis_title="Monitoring Year",
        yaxis_title=f"{parameter} (Median)",
        height=360
    )
    fig.update_xaxes(tickmode='linear', tick0=2003, dtick=1, gridcolor=COLOR_GRID)
    return fig

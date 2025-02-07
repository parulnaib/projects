import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Risk Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Risk Analytics Dashboard")

# Generate sample data
def generate_sample_data():
    import numpy as np
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
    data = []

    for asset in assets:
        base_var = np.random.uniform(2000, 4000)
        base_sharpe = np.random.uniform(0.5, 1.5)
        base_vol = np.random.uniform(0.2, 0.4)

        for date in dates:
            data.append({
                'date': date,
                'asset': asset,
                'value_at_risk': base_var + np.random.uniform(-200, 200),
                'sharpe_ratio': base_sharpe + np.random.uniform(-0.1, 0.1),
                'volatility': base_vol + np.random.uniform(-0.05, 0.05)
            })

    return pd.DataFrame(data)

# Load data
df = generate_sample_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("---")

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    options=df['asset'].unique(),
    default=df['asset'].unique()[:3]
)

# Date range filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Filter data
filtered_df = df[
    (df['asset'].isin(selected_assets)) &
    (df['date'].dt.date >= date_range[0]) &
    (df['date'].dt.date <= date_range[1])
]

# Create metrics summary
st.markdown("### 📈 Key Risk Metrics")
metrics_cols = st.columns(3)

with metrics_cols[0]:
    avg_var = filtered_df['value_at_risk'].mean()
    st.metric("Average VaR", f"${avg_var:,.2f}")

with metrics_cols[1]:
    avg_sharpe = filtered_df['sharpe_ratio'].mean()
    st.metric("Average Sharpe Ratio", f"{avg_sharpe:.2f}")

with metrics_cols[2]:
    avg_vol = filtered_df['volatility'].mean()
    st.metric("Average Volatility", f"{avg_vol:.2%}")

st.markdown("---")

# Create two columns for charts
col1, col2 = st.columns(2)

# Value at Risk Chart
with col1:
    st.subheader("Value at Risk Over Time")
    fig_var = go.Figure()

    for asset in selected_assets:
        asset_data = filtered_df[filtered_df['asset'] == asset]
        fig_var.add_trace(go.Scatter(
            x=asset_data['date'],
            y=asset_data['value_at_risk'],
            name=asset,
            mode='lines',
            line=dict(width=2)
        ))

    fig_var.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Value at Risk ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_var, use_container_width=True)

# Sharpe Ratio Chart
with col2:
    st.subheader("Sharpe Ratio Comparison")
    fig_sharpe = go.Figure()

    for asset in selected_assets:
        asset_data = filtered_df[filtered_df['asset'] == asset]
        fig_sharpe.add_trace(go.Scatter(
            x=asset_data['date'],
            y=asset_data['sharpe_ratio'],
            name=asset,
            mode='lines',
            line=dict(width=2)
        ))

    fig_sharpe.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

# Volatility Chart
st.subheader("Volatility Analysis")
fig_vol = go.Figure()

for asset in selected_assets:
    asset_data = filtered_df[filtered_df['asset'] == asset]
    fig_vol.add_trace(go.Scatter(
        x=asset_data['date'],
        y=asset_data['volatility'],
        name=asset,
        mode='lines',
        line=dict(width=2)
    ))

fig_vol.update_layout(
    height=400,
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="Volatility (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_tickformat='.2%'
)
st.plotly_chart(fig_vol, use_container_width=True)

# Metrics Table
st.markdown("### 📊 Latest Risk Metrics")
latest_metrics = filtered_df.groupby('asset').last().reset_index()[['asset', 'value_at_risk', 'sharpe_ratio', 'volatility']]
latest_metrics['volatility'] = latest_metrics['volatility'].map('{:.2%}'.format)
latest_metrics['value_at_risk'] = latest_metrics['value_at_risk'].map('${:,.2f}'.format)
latest_metrics['sharpe_ratio'] = latest_metrics['sharpe_ratio'].map('{:.2f}'.format)
latest_metrics.columns = ['Asset', 'Value at Risk', 'Sharpe Ratio', 'Volatility']
st.dataframe(latest_metrics, use_container_width=True)

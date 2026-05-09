import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    at_risk    = pd.read_csv('data/full_at_risk_customers.csv')
    revenue_df = pd.read_csv('data/full_at_risk_customers.csv')
    roi        = json.load(open('data/roi_summary.json'))
    return at_risk, revenue_df, roi

@st.cache_resource
def load_model():
    model         = pickle.load(open('models/xgb_tuned.pkl', 'rb'))
    feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))
    return model, feature_names

at_risk_df, revenue_df, roi = load_data()
model, feature_names        = load_model()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📈 Overview", "🔍 Risk Explorer", "💰 Revenue Impact", "🤖 Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost (tuned)")
st.sidebar.markdown(f"**AUC-ROC:** 0.871")
st.sidebar.markdown(f"**Threshold:** 0.35")
st.sidebar.markdown("**Data:** IBM Telco Dataset")

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────
if page == "📈 Overview":
    st.title("📈 Customer Churn Overview")
    st.markdown("Real-time churn risk monitoring and revenue impact summary.")
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Scored",
                f"{roi['total_customers_scored']:,}")
    col2.metric("At-Risk Customers",
                f"{roi['customers_at_risk']:,}",
                f"{roi['customers_at_risk']/roi['total_customers_scored']*100:.1f}% of base")
    col3.metric("Monthly Revenue at Risk",
                f"${roi['monthly_revenue_at_risk']:,.0f}",
                "Needs retention action", delta_color="inverse")
    col4.metric("Retention Program ROI",
                f"{roi['retention_roi_pct']:.0f}%",
                f"${roi['monthly_net_gain']:,.0f} net gain/mo")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Churn probability distribution
        fig = px.histogram(
            at_risk_df, x='churn_probability', nbins=30,
            title='Churn Probability Distribution (At-Risk Customers)',
            color_discrete_sequence=['#D85A30'],
            labels={'churn_probability': 'Churn Probability'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Risk segment pie
        segment_counts = at_risk_df['risk_level'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig2 = px.pie(
            segment_counts, values='Count', names='Segment',
            title='At-Risk Customers by Segment',
            color_discrete_sequence=['#D85A30','#F5A623','#378ADD','#AAAAAA']
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Churn by contract type bar
    contract_data = at_risk_df.groupby('contract').agg(
        Customers=('monthly_charges','count'),
        Monthly_Revenue=('monthly_charges','sum')
    ).reset_index()

    fig3 = px.bar(
        contract_data, x='contract', y='Monthly_Revenue',
        title='Monthly Revenue at Risk by Contract Type',
        color='contract',
        color_discrete_sequence=['#D85A30','#F5A623','#378ADD'],
        labels={'Monthly_Revenue': 'Monthly Revenue ($)', 'contract': 'Contract Type'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2: RISK EXPLORER
# ─────────────────────────────────────────
elif page == "🔍 Risk Explorer":
    st.title("🔍 At-Risk Customer Explorer")
    st.markdown("Filter and explore the full ranked list of at-risk customers.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_prob = st.slider("Min Churn Probability",
                             0.0, 1.0, float(THRESHOLD if 'THRESHOLD' in dir() else 0.35),
                             0.05)
    with col2:
        contract_filter = st.multiselect(
            "Contract Type",
            options=at_risk_df['contract'].unique().tolist(),
            default=at_risk_df['contract'].unique().tolist()
        )
    with col3:
        min_clv = st.number_input("Min CLV ($)", value=0, step=100)

    filtered = at_risk_df[
        (at_risk_df['churn_probability'] >= min_prob) &
        (at_risk_df['contract'].isin(contract_filter)) &
        (at_risk_df['clv'] >= min_clv)
    ].sort_values('clv', ascending=False)

    st.markdown(f"**{len(filtered):,} customers** match your filters "
                f"| Total CLV at risk: **${filtered['clv'].sum():,.0f}**")

    # Risk × Value scatter
    fig = px.scatter(
        filtered,
        x='churn_probability', y='clv',
        color='risk_level', size='monthly_charges',
        hover_data=['contract', 'tenure', 'monthly_charges'],
        title='Risk × Value Matrix (filtered)',
        labels={'churn_probability': 'Churn Probability',
                'clv': 'Customer Lifetime Value ($)'},
        color_discrete_map={
            'Critical' : '#D85A30',
    'High'     : '#F5A623',
    'Medium'   : '#378ADD',
    'Low'      : '#AAAAAA'
        }
    )
    fig.add_vline(x=0.5, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=filtered['clv'].median(),
                  line_dash='dash', line_color='gray', opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Customer List — Ranked by CLV")
    display_cols = ['rank', 'churn_probability', 'risk_level',
                    'monthly_charges', 'clv', 'tenure', 'contract']
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].head(50).style.background_gradient(
            subset=['churn_probability'], cmap='Reds'
        ).background_gradient(
            subset=['clv'], cmap='Blues'
        ).format({
            'churn_probability': '{:.1%}',
            'monthly_charges'  : '${:.0f}',
            'clv'              : '${:,.0f}'
        }),
        use_container_width=True
    )

    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download filtered list as CSV",
        csv,
        "at_risk_customers.csv",
        "text/csv"
    )

# ─────────────────────────────────────────
# PAGE 3: REVENUE IMPACT
# ─────────────────────────────────────────
elif page == "💰 Revenue Impact":
    st.title("💰 Revenue Impact & Retention ROI")
    st.markdown("Model the financial case for your retention program.")
    st.markdown("---")

    st.subheader("Retention Program Simulator")
    st.markdown("Adjust the parameters to model different retention scenarios.")

    col1, col2, col3 = st.columns(3)
    with col1:
        contact_n     = st.slider("Customers to contact per month",
                                  50, 500, 200, 25)
    with col2:
        offer_cost    = st.slider("Cost per outreach ($)",
                                  10, 200, 50, 10)
    with col3:
        success_rate  = st.slider("Retention success rate (%)",
                                  5, 60, 30, 5) / 100

    # Calculate
    top_n             = at_risk_df.sort_values('clv', ascending=False).head(contact_n)
    total_cost        = contact_n * offer_cost
    customers_saved   = int(contact_n * success_rate)
    rev_saved         = top_n.head(customers_saved)['monthly_charges'].sum()
    clv_saved         = top_n.head(customers_saved)['clv'].sum()
    net_gain          = rev_saved - total_cost
    roi_calc          = (net_gain / total_cost) * 100 if total_cost > 0 else 0

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outreach Cost", f"${total_cost:,}")
    c2.metric("Customers Saved", f"~{customers_saved}")
    c3.metric("Monthly Revenue Saved", f"${rev_saved:,.0f}")
    c4.metric("Net Monthly Gain",
              f"${net_gain:,.0f}",
              f"ROI: {roi_calc:.0f}%",
              delta_color="normal" if net_gain > 0 else "inverse")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        # Cost vs revenue bar
        fig = go.Figure(data=[
            go.Bar(name='Outreach Cost',
                   x=['Program Financials'],
                   y=[total_cost],
                   marker_color='#D85A30'),
            go.Bar(name='Revenue Saved',
                   x=['Program Financials'],
                   y=[rev_saved],
                   marker_color='#378ADD'),
            go.Bar(name='Net Gain',
                   x=['Program Financials'],
                   y=[net_gain],
                   marker_color='#1D9E75'),
        ])
        fig.update_layout(title='Cost vs Revenue Saved',
                          barmode='group', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Sensitivity curve
        rates  = np.arange(0.05, 0.65, 0.05)
        gains  = []
        for r in rates:
            s   = int(contact_n * r)
            rev = top_n.head(s)['monthly_charges'].sum()
            gains.append(rev - total_cost)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rates*100, y=gains,
            mode='lines+markers',
            line=dict(color='#378ADD', width=2.5),
            name='Net Gain'
        ))
        fig2.add_hline(y=0, line_dash='dash',
                       line_color='#D85A30',
                       annotation_text='Break-even')
        fig2.update_layout(
            title='ROI Sensitivity Analysis',
            xaxis_title='Retention Success Rate (%)',
            yaxis_title='Net Monthly Gain ($)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"💡 At a {success_rate*100:.0f}% retention rate, "
            f"every **$1 spent** on this program returns "
            f"**${1 + roi_calc/100:.1f}** in saved revenue.")

# ─────────────────────────────────────────
# PAGE 4: MODEL INSIGHTS
# ─────────────────────────────────────────
elif page == "🤖 Model Insights":
    st.title("🤖 Model Insights & Explainability")
    st.markdown("Understand what drives churn predictions.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        metrics = {
            'Metric'   : ['AUC-ROC', 'Recall', 'Precision', 'F1 Score'],
            'Score'    : [0.871, 0.762, 0.681, 0.719],
            'Target'   : ['≥ 0.85 ✅', '≥ 75% ✅', '≥ 65% ✅', '—']
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

        st.markdown("---")
        st.subheader("Why not just use accuracy?")
        st.markdown("""
        With 26.5% churn rate, a model that predicts **"no churn" for everyone**
        would be **73.5% accurate** — but completely useless.

        We optimise for **Recall** (catching real churners) and
        **AUC-ROC** (overall discrimination power) instead.
        """)

    with col2:
        st.subheader("Key Churn Drivers")
        st.markdown("Top features pushing customers toward churn:")

        drivers = pd.DataFrame({
            'Feature'   : ['Month-to-month contract',
                           'High charges per tenure',
                           'Short tenure (< 6 months)',
                           'High monthly charges',
                           'No online security',
                           'No tech support',
                           'Fiber optic internet',
                           'Electronic check payment'],
            'Impact'    : [0.31, 0.24, 0.19, 0.16, 0.14, 0.12, 0.10, 0.09],
            'Direction' : ['↑ Churn'] * 8
        })

        fig = px.bar(
            drivers.sort_values('Impact'),
            x='Impact', y='Feature',
            orientation='h',
            title='Top Churn Drivers (SHAP importance)',
            color_discrete_sequence=['#D85A30']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Charts")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Load SHAP Summary Plot"):
            st.image('charts/11_shap_global.png',
                     caption='Global Feature Importance (SHAP)',
                     use_column_width=True)
    with col_b:
        if st.button("Load SHAP Beeswarm Plot"):
            st.image('charts/12_shap_beeswarm.png',
                     caption='Feature Impact Direction (SHAP)',
                     use_column_width=True)
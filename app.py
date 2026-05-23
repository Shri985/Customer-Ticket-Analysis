"""
app.py
Streamlit dashboard for Customer Ticket Analysis and Auto Response System.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from predict import TicketAnalyzer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Ticket AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .priority-high { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
    .priority-medium { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .priority-low { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .response-box {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 5px; font-family: monospace;
        white-space: pre-wrap; font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI models...")
def load_analyzer():
    return TicketAnalyzer()


@st.cache_data
def load_dataset():
    if os.path.exists("data/customer_tickets.csv"):
        return pd.read_csv("data/customer_tickets.csv")
    return None


def priority_badge(priority):
    colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
    color = colors.get(priority, "#95a5a6")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem">{priority}</span>'


def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/customer-support.png", width=80)
    st.sidebar.title("🎫 Ticket AI System")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    page = st.sidebar.radio("", ["🔍 Analyze Ticket", "📊 Analytics Dashboard", "📈 Model Performance"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.info(
        "AI-powered customer ticket classification and auto-response system. "
        "Uses ensemble ML (LightGBM + XGBoost + Logistic Regression) for high accuracy."
    )
    return page


def page_analyze(analyzer):
    st.title("🔍 Customer Ticket Analyzer")
    st.markdown("Enter a customer support ticket to get instant AI-powered classification and auto-response.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Ticket Input")
        subject = st.text_input("Subject Line", placeholder="e.g., Charged twice this month")
        ticket_text = st.text_area(
            "Ticket Description",
            placeholder="Describe the customer's issue in detail...",
            height=150,
        )

        with st.expander("⚙️ Additional Context (Optional)"):
            c1, c2, c3 = st.columns(3)
            customer_age = c1.number_input("Customer Age (days)", 1, 3650, 365)
            prev_tickets = c2.number_input("Previous Tickets", 0, 50, 0)
            channel = c3.selectbox("Channel", ["Web Form", "Email", "Chat", "Phone"])

        analyze_btn = st.button("🚀 Analyze Ticket", type="primary", use_container_width=True)

    with col2:
        st.subheader("💡 Sample Tickets")
        samples = {
            "Billing Issue (Urgent)": (
                "Charged twice this month",
                "I was charged twice for my subscription this month. This is extremely urgent and needs immediate attention! I need a refund immediately."
            ),
            "Technical Problem": (
                "App keeps crashing",
                "The mobile app crashes every time I open it after the latest update. I cannot use the service at all."
            ),
            "Delivery Issue": (
                "Order not delivered",
                "My order was supposed to arrive 3 days ago but I still haven't received it. The tracking shows delivered but it wasn't."
            ),
            "Account Access": (
                "Cannot login to account",
                "I cannot log in to my account. It says invalid credentials but I haven't changed my password. Please help."
            ),
        }
        for label, (subj, text) in samples.items():
            if st.button(f"📋 {label}", use_container_width=True):
                st.session_state["sample_subject"] = subj
                st.session_state["sample_text"] = text
                st.rerun()

        if "sample_subject" in st.session_state:
            st.info(f"**Subject:** {st.session_state['sample_subject']}\n\n**Text:** {st.session_state['sample_text'][:100]}...")

    # Use sample if loaded
    if "sample_subject" in st.session_state and not subject:
        subject = st.session_state.pop("sample_subject")
        ticket_text = st.session_state.pop("sample_text")

    if analyze_btn and subject and ticket_text:
        with st.spinner("🤖 Analyzing ticket..."):
            result = analyzer.predict(subject, ticket_text, customer_age, prev_tickets, channel)

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏷️ Category", result["category"])
        m2.metric("😊 Sentiment", result["sentiment"].capitalize())
        m3.metric("⚡ Priority", result["priority"])
        m4.metric("🎯 Confidence", f"{result['category_confidence']}%")

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("📈 Category Confidence Breakdown")
            breakdown = result["category_breakdown"]
            fig = px.bar(
                x=list(breakdown.values()),
                y=list(breakdown.keys()),
                orientation="h",
                color=list(breakdown.values()),
                color_continuous_scale="Viridis",
                labels={"x": "Confidence (%)", "y": "Category"},
            )
            fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Ticket ID | `{result['ticket_id']}` |
            | Priority | {priority_color.get(result['priority'], '⚪')} {result['priority']} |
            | Urgency Score | {result['urgency_score']}/10 |
            | Sentiment Confidence | {result['sentiment_confidence']}% |
            """)

        with col_b:
            st.subheader("📧 Auto-Generated Response")
            st.markdown(
                f'<div class="response-box">{result["auto_response"]}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Response",
                result["auto_response"],
                file_name=f"{result['ticket_id']}_response.txt",
                mime="text/plain",
                use_container_width=True,
            )

    elif analyze_btn:
        st.warning("Please fill in both Subject and Ticket Description.")


def page_analytics():
    st.title("📊 Analytics Dashboard")
    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Run `python generate_dataset.py` first.")
        return

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Tickets", f"{len(df):,}")
    k2.metric("Categories", df["category"].nunique())
    k3.metric("High Priority", f"{(df['priority']=='High').sum():,}")
    k4.metric("Avg Ticket Length", f"{df['ticket_text'].str.len().mean():.0f} chars")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tickets by Category")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Priority Distribution")
        pri_counts = df["priority"].value_counts().reset_index()
        pri_counts.columns = ["Priority", "Count"]
        colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
        fig = px.bar(pri_counts, x="Priority", y="Count",
                     color="Priority", color_discrete_map=colors)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Sentiment by Category")
        pivot = df.groupby(["category", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(pivot, x="category", y="count", color="sentiment",
                     barmode="group",
                     color_discrete_map={"urgent": "#e74c3c", "normal": "#3498db", "low": "#27ae60"})
        fig.update_layout(height=350, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Channel Distribution")
        ch_counts = df["channel"].value_counts().reset_index()
        ch_counts.columns = ["Channel", "Count"]
        fig = px.funnel(ch_counts, x="Count", y="Channel",
                        color_discrete_sequence=["#667eea"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Sample Tickets")
    st.dataframe(
        df[["ticket_id", "subject", "category", "sentiment", "priority", "channel"]].head(20),
        use_container_width=True,
    )


def page_performance():
    st.title("📈 Model Performance")
    results_path = "reports/model_results.json"
    if not os.path.exists(results_path):
        st.warning("Model not trained yet. Run `python train.py` first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    st.subheader("🏆 Model Accuracy Summary")
    cols = st.columns(len(results))
    for col, (task, metrics) in zip(cols, results.items()):
        col.metric(
            f"{task.capitalize()} Accuracy",
            f"{metrics['accuracy']*100:.2f}%",
            f"F1: {metrics['f1']:.4f}",
        )

    st.markdown("---")
    st.subheader("📊 Accuracy Comparison")
    tasks = list(results.keys())
    accuracies = [results[t]["accuracy"] * 100 for t in tasks]
    f1_scores = [results[t]["f1"] * 100 for t in tasks]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy (%)", x=tasks, y=accuracies,
                         marker_color="#667eea"))
    fig.add_trace(go.Bar(name="F1 Score (%)", x=tasks, y=f1_scores,
                         marker_color="#764ba2"))
    fig.update_layout(barmode="group", height=400, yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🖼️ Confusion Matrices")
    report_dir = "reports"
    img_files = [f for f in os.listdir(report_dir) if f.endswith("_confusion_matrix.png")]
    if img_files:
        img_cols = st.columns(len(img_files))
        for col, img_file in zip(img_cols, img_files):
            col.image(
                os.path.join(report_dir, img_file),
                caption=img_file.replace("_confusion_matrix.png", "").replace("_", " ").title(),
                width=600,
            )
    else:
        st.info("Confusion matrix images will appear here after training.")

    st.subheader("🔧 Model Architecture")
    st.markdown("""
    | Component | Details |
    |-----------|---------|
    | **Category Model** | Voting Ensemble: LightGBM (w=2) + XGBoost (w=2) + Logistic Regression (w=1) |
    | **Sentiment Model** | LightGBM Classifier (300 estimators) |
    | **Priority Model** | LightGBM Classifier (300 estimators) |
    | **Feature Extraction** | TF-IDF (15K features, 1-3 ngrams) + 8 engineered features |
    | **Text Preprocessing** | Lemmatization, stopword removal, urgency keyword scoring |
    | **Class Balancing** | class_weight="balanced" in all models |
    | **Evaluation** | Stratified 80/20 split + 5-fold cross-validation |
    """)


def main():
    page = render_sidebar()
    try:
        analyzer = load_analyzer()
    except Exception:
        analyzer = None
        if "Analyze" in page:
            st.error("⚠️ Models not found. Please run: `python train.py`")

    if "Analyze" in page:
        if analyzer:
            page_analyze(analyzer)
        else:
            st.info("Train the model first using `python train.py`")
    elif "Analytics" in page:
        page_analytics()
    elif "Performance" in page:
        page_performance()


if __name__ == "__main__":
    main()

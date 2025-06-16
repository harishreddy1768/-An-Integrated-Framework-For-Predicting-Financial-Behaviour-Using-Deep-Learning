import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and preprocessor
xgb_model = joblib.load("model_xgb.pkl")
ann_model = load_model("model_ann.h5")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Smart Financial Recommender & Spending Analyzer ", layout="centered")

# -----------------------
# Custom Styling & Logo
# -----------------------
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 6px 12px;
    }
    .recommend-card {
        background-color: #f0f8ff;
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #1e90ff;
        margin-bottom: 20px;
        font-size: 15px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: gray;
        margin-top: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Smart Financial Recommender & Spending Analyzer ")
st.markdown("A smart way to analyze your expenses and receive personal financial advice.")

# ------------------------
# Input Form
# ------------------------
st.subheader("Enter Your Financial Details")

labels = [
    "Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries",
    "Transport", "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"
]

with st.form("input_form"):
    cols = st.columns(2)
    user_input = {}
    for i, label in enumerate(labels):
        with cols[i % 2]:
            user_input[label] = st.number_input(f"{label}", min_value=0, step=100, format="%d", value=0)
    submitted = st.form_submit_button("Submit")

# ------------------------
# Derived Fields
# ------------------------
if submitted:
    total_expense = sum(user_input[label] for label in labels[3:])
    disposable_income = user_input["Income"] - total_expense
    desired_savings = disposable_income * 0.3
    desired_savings_percentage = (desired_savings / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0

    input_df = pd.DataFrame([user_input])
    input_df["Disposable_Income"] = disposable_income
    input_df["Desired_Savings"] = desired_savings
    input_df["Desired_Savings_Percentage"] = desired_savings_percentage

    input_df = input_df[preprocessor.feature_names_in_]
    input_processed = preprocessor.transform(input_df)

    xgb_pred = xgb_model.predict(input_processed)[0]
    ann_pred = np.argmax(ann_model.predict(input_processed), axis=1)[0]

    label_map = {0: "💸 Heavy Spender", 1: "💼 Moderate Spender", 2: "💰 Saver"}
    final_label = label_map[xgb_pred]

    # ------------------------
    # Tabs
    # ------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🧠 Predictions", "📌 Recommendations"])

    with tab1:
        st.markdown("### 📌 Expense Distribution")
        fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
        pie_data = [user_input[l] for l in labels[3:]]
        ax1.pie(pie_data, labels=labels[3:], autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        st.markdown("### 📊 Category Breakdown")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=labels[3:], y=pie_data, palette="coolwarm", ax=ax2)
        ax2.set_ylabel("Amount (₹)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with tab2:
        st.success(f"### Predicted Category: **{final_label}**")
        st.metric("💵 Disposable Income", f"₹{disposable_income:,.0f}")
        st.metric("💸 Desired Savings", f"₹{desired_savings:,.0f}")
        st.metric("📈 Desired Savings %", f"{desired_savings_percentage:.2f}%")

    with tab3:
        st.markdown(f"<div class='recommend-card'><h4>💡 Personal Advice for {final_label}</h4>", unsafe_allow_html=True)

        if xgb_pred == 0:
            st.markdown(f"""
            - Reduce **entertainment** and **eating out** expenses.
            - Set up a savings goal of ₹{desired_savings:,.0f} per month.
            - Use an expense tracker to stay on budget.
            """)
        elif xgb_pred == 1:
            st.markdown(f"""
            - Moderate spender. Try trimming **utilities** or **transport** costs.
            - Consider SIPs or fixed deposits for better savings.
            """)
        else:
            st.markdown(f"""
            - You're doing well! Maintain or increase your savings rate.
            - Explore investment opportunities (e.g., mutual funds, PPF).
            - Build an emergency fund if not already.
            """)

        st.markdown("""
        <h4>📘 Financial Guidelines</h4>
        - 💳 Keep debt below 36% of income  
        - 💡 Follow 50/30/20 rule (Needs/Wants/Savings)  
        - 📈 Review spending monthly  
        - 💰 Automate savings where possible  
        </div>
        """, unsafe_allow_html=True)

# ------------------------
# Footer with Your Name
# ------------------------
st.markdown("<div class='footer'>Built by <b>Harish Reddy Singireddy </b> 🌟</div>", unsafe_allow_html=True)

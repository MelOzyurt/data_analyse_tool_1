import streamlit as st
import pandas as pd
import numpy as np
import re
import openai
from utils.analysis_utils import analyze_numeric, correlation_plot, chi_square_analysis, t_test_analysis

# ✅ OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def ai_interpretation(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant. Provide insight and explain statistical results in plain language."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"**AI Error:** {e}"

def analyze_data_ai_ui(df: pd.DataFrame):
    st.header("🔍 AI-Assisted Data Analysis")

    if df is None or df.empty:
        st.warning("No data uploaded or empty DataFrame.")
        return

    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary",
        "Correlation Matrix",
        "Chi-Square Test",
        "T-Test"
    ])

    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write("### 📊 Descriptive Statistics")
        st.dataframe(result)

        prompt = f"Analyze the following numeric summary statistics:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### 🧠 AI Insight")
        st.write(ai_result)

    elif option == "Correlation Matrix":
        st.write("### 📈 Correlation Matrix")
        fig, corr_df = correlation_plot(df)
        st.plotly_chart(fig, use_container_width=True)

        prompt = f"Explain the key findings in this correlation matrix:\n{corr_df.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### 🧠 AI Insight")
        st.write(ai_result)

    elif option == "Chi-Square Test":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols) < 2:
            st.error("Not enough categorical columns for Chi-Square Test.")
            return

        col1 = st.selectbox("Select first categorical column", cat_cols)
        col2 = st.selectbox("Select second categorical column", [c for c in cat_cols if c != col1])

        result, p_val = chi_square_analysis(df, col1, col2)
        st.write(f"**Chi² = {result['chi2_stat']:.2f}, p = {result['p_value']:.4f}**")
        st.dataframe(result["contingency_table"])

        prompt = f"Interpret the chi-square result with p={p_val} between {col1} and {col2}."
        ai_result = ai_interpretation(prompt)
        st.markdown("### 🧠 AI Insight")
        st.write(ai_result)

    elif option == "T-Test":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            st.error("Not enough numeric columns for T-Test.")
            return

        col1 = st.selectbox("Select first numeric column", num_cols)
        col2 = st.selectbox("Select second numeric column", [c for c in num_cols if c != col1])

        try:
            result, p_val = t_test_analysis(df, col1, col2)
            st.write(result)
            prompt = f"Interpret the t-test result with p={p_val} comparing {col1} and {col2}."
            ai_result = ai_interpretation(prompt)
            st.markdown("### 🧠 AI Insight")
            st.write(ai_result)
        except Exception as e:
            st.error(f"T-Test Error: {e}")

    st.success("✅ Analysis completed.")

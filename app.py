#gpt-4.1 version

import streamlit as st
import openai
import pandas as pd
import numpy as np
from analysis_utils import *
from utils_text import *

# Başlık
st.title("🧠 Data Analysis Tool")

# OpenAI API Key
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# GPT interpret function (with updated API)
def ai_interpretation(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # gpt-4.1 uyumlu
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights. You can highlighted the anomalies, interpretcorrelations between atributes, find and tell similarities or impact from other attributes"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI yorumlama sırasında hata oluştu:\n\n{e}"

# Veri yükleme
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary",
        "Correlation Matrix",
        "Chi-Square Test",
        "T-Test",
        "Custom Analysis"
    ])

    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write(result)

        # AI Yorumu
        prompt = f"Analyze the following numeric summary statistics and provide insights:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

    elif option == "Correlation Matrix":
        fig, corr_df = correlation_plot(df)
        st.plotly_chart(fig)

        prompt = f"Explain the key points and findings from this correlation matrix:\n{corr_df.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

    elif option == "Chi-Square Test":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) < 2:
            st.error("Dataset does not have enough categorical columns for Chi-Square test.")
        else:
            col1 = st.selectbox("Select first categorical column", categorical_cols)
            col2 = st.selectbox("Select second categorical column", categorical_cols, index=1 if len(categorical_cols) > 1 else 0)
            if col1 and col2 and col1 != col2:
                result, p_val = chi_square_analysis(df, col1, col2)
                st.write(result)
                prompt = f"Interpret the chi-square test result with p-value {p_val} between {col1} and {col2}."
                ai_result = ai_interpretation(prompt)
                st.markdown("### AI Insights")
                st.write(ai_result)

    elif option == "T-Test":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset does not have enough numeric columns for T-Test.")
        else:
            col1 = st.selectbox("Select first numeric column", numeric_cols)
            col2 = st.selectbox("Select second numeric column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            if col1 and col2 and col1 != col2:
                result, p_val = t_test_analysis(df, col1, col2)
                st.write(result)
                prompt = f"Interpret the t-test result with p-value {p_val} comparing {col1} and {col2}."
                ai_result = ai_interpretation(prompt)
                st.markdown("### AI Insights")
                st.write(ai_result)

    elif option == "Custom Analysis":
        result = custom_analysis(df)
        st.write(result)
        prompt = f"Provide insights for this custom analysis:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

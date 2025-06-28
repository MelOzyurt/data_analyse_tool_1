import streamlit as st
import pandas as pd
import numpy as np
import re
import openai
from analysis_utils import *
from utils_text import *
from analysis_utils import t_test_analysis

# ✅ Application configration
st.set_page_config(page_title="📊 Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")

# ✅ OpenAI istemcisi
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ AI Interpretation
def ai_interpretation(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights. You can highlight anomalies, interpret correlations between attributes, find and tell similarities or impact from other attributes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        raw_message = response.choices[0].message.content.strip()
        sentences = re.findall(r'[^.!?]*[.!?]', raw_message)
        return ''.join(sentences).strip()
    except Exception as e:
        return f"**Error during AI interpretation:** {e}"

# ✅ Uploading File
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, XML, Feather)",
    type=["csv", "xlsx", "xls", "json", "xml", "feather"]
)

if uploaded_file:
    try:
        # File Typles
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # ✅ Data Priview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ✅ Choice the analysis
    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary",
        "Correlation Matrix",
        "Chi-Square Test",
        "T-Test"
    ])

    # ✅ Numerical Analsis Summary
    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write("### 📊 Descriptive Statistics")
        st.dataframe(result)

        # AI Interpret
        prompt = f"Analyze the following numeric summary statistics and provide insights:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### 🧠 AI Insights")
        st.write(ai_result)

    # ✅ Correlation Matrix
    elif option == "Correlation Matrix":
        st.write("### 📈 Correlation Matrix")
        fig, corr_df = correlation_plot(df)
        st.plotly_chart(fig, use_container_width=True)

        prompt = f"Explain the key points and findings from this correlation matrix:\n{corr_df.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### 🧠 AI Insights")
        st.write(ai_result)

    # ✅ Chi-Square Test
    elif option == "Chi-Square Test":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) >= 2:
            col1 = st.selectbox("Select first categorical column", categorical_cols)
            col2_options = [c for c in categorical_cols if c != col1]
            col2 = st.selectbox("Select second categorical column", col2_options)

            result, p_val = chi_square_analysis(df, col1, col2)
            st.markdown(f"**Chi-Square Test Result:** χ² = {result['chi2_stat']:.2f}, p = {result['p_value']:.4f}")
            st.dataframe(result["contingency_table"])

            prompt = f"Interpret the chi-square test result with p-value {p_val} between {col1} and {col2}."
            ai_result = ai_interpretation(prompt)
            st.markdown("###  Data Insights")
            st.write(ai_result)
        else:
            st.error("Dataset does not have enough categorical columns for Chi-Square test.")

    # ✅ T-Test
    elif option == "T-Test":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Select first numeric column", numeric_cols)
            col2 = st.selectbox("Select second numeric column", [c for c in numeric_cols if c != col1])

            try:
                result, p_val = t_test_analysis(df, col1, col2)
                st.write(result)
                prompt = f"Interpret the t-test result with p-value {p_val} comparing {col1} and {col2}."
                ai_result = ai_interpretation(prompt)
                st.markdown("### 🧠 AI Insights")
                st.write(ai_result)
            except Exception as e:
                st.error(f"T-Test Error: {e}")
        else:
            st.error("Dataset does not have enough numeric columns for T-Test.")

    st.success("✅ Analysis completed successfully!")

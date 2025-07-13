import streamlit as st
import pandas as pd

# App Modules
from auth.login import login_ui
from components.sidebar import show_sidebar
from upload.file_handler import upload_and_process_file
from analysis.analyze_data import analyze_data_ui
from analysis.analyze_feedback import analyze_feedback_ui
from chat.chat_with_doc import chat_with_doc_ui
from components.cost_panel import show_cost_panel

# Page config
st.set_page_config(page_title="📊 Unified Analytics App", layout="wide")

# --- Session Initialization ---
def init_session():
    defaults = {
        "logged_in": False,
        "username": None,
        "balance": 50.0,  # Default balance
        "uploaded_data": None,
        "analysis_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# --- Main App Function ---
def main():
    st.title("📈 Unified Analytics Platform")

    # Sidebar (Login info & logout)
    app_mode = show_sidebar()

    # Login Control
    if not st.session_state.get("logged_in", False):
        login_ui()
        return

    # Upload Section
    st.header("📁 Upload Your Data")
    uploaded_data, file_type = upload_and_process_file()

    if uploaded_data is None:
        st.info("Please upload a file to proceed.")
        return

    st.success(f"✅ File uploaded successfully! Detected type: `{file_type}`")
    st.dataframe(uploaded_data.head())
    st.session_state.uploaded_data = uploaded_data

    # Analysis Type Selection
    st.subheader("🔎 Choose Analysis Type")
    analysis_type = st.radio(
        "Select one of the following analysis options:",
        ["AI-Assisted Data Analysis", "Analyze Feedback", "Chat With Document"]
    )
    st.session_state.analysis_type = analysis_type

    # Cost Panel
    show_cost_panel(analysis_type)

    # Run Analysis
    if st.button("🚀 Run Analysis"):
        if st.session_state.balance <= 0:
            st.error("❌ Insufficient balance. Please top up your account.")
            return

        if analysis_type == "AI-Assisted Data Analysis":
            analyze_data_ui(uploaded_data)
        elif analysis_type == "Analyze Feedback":
            analyze_feedback_ui(uploaded_data)
        elif analysis_type == "Chat With Document":
            chat_with_doc_ui(uploaded_data)
        else:
            st.warning("⚠️ Invalid selection.")

# --- App Entry Point ---
if __name__ == "__main__":
    main()

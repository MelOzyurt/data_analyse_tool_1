import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind

# ---------- Load Data ----------
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

# ---------- 1. Numeric Summary ----------
def analyze_numeric(df):
    numeric_df = df.select_dtypes(include=np.number)
    summary = numeric_df.describe().transpose()
    return summary

# ---------- 2. Correlation Plot ----------
def correlation_plot(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    return fig, corr

# ---------- 3. Chi-Square Test ----------
def chi_square_analysis(df, col1=None, col2=None):
    if col1 is None or col2 is None:
        # Just return column names so user can pick
        return {"error": "You must specify two categorical columns."}, None

    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    results = pd.DataFrame({
        "Chi² Statistic": [chi2],
        "Degrees of Freedom": [dof],
        "P-Value": [p]
    })

    return results, p

#-------------4. T-Test Analaysis-----------------------

import pandas as pd
from scipy.stats import ttest_ind

def t_test_analysis(df, col1, col2):
    if not pd.api.types.is_numeric_dtype(df[col1]):
        return {"error": f"{col1} must be numeric"}, None
    if not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"{col2} must be numeric"}, None

    stat, p = ttest_ind(df[col1].dropna(), df[col2].dropna(), equal_var=False)
    result = pd.DataFrame({
        "T-Statistic": [stat],
        "P-Value": [p]
    })
    return result, p

#-------------5. Custom Analysis----------------------

def custom_analysis(df):
    try:
        summary = df.describe(include='all')
        missing = df.isnull().sum()
        result = f"Data Summary:\n{summary}\n\nMissing Values:\n{missing}"
        return result
    except Exception as e:
        return f"Error in custom analysis: {e}"

 

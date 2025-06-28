import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats

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
from scipy.stats import ttest_ind

def t_test_analysis(df, col1, col2):
    if df[col1].dtype != 'float64' and df[col1].dtype != 'int64':
        return {"error": f"{col1} must be numeric"}, None
    if df[col2].dtype != 'float64' and df[col2].dtype != 'int64':
        return {"error": f"{col2} must be numeric"}, None

    stat, p = ttest_ind(df[col1].dropna(), df[col2].dropna(), equal_var=False)
    result = pd.DataFrame({
        "T-Statistic": [stat],
        "P-Value": [p]
    })
    return result, p
def custom_analysis(df):
    return df.head()  

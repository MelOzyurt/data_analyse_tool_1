import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats

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

# ---------- 4. Additional Example Analysis: T-test for two numeric columns ----------
def t_test_analysis(df, col1=None, col2=None):
    if col1 is None or col2 is None:
        return {"error": "You must specify two numeric columns."}, None

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "Columns not found in dataframe."}, None

    if not np.issubdtype(df[col1].dtype, np.number) or not np.issubdtype(df[col2].dtype, np.number):
        return {"error": "Selected columns must be numeric."}, None

    stat, p = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
    results = pd.DataFrame({
        "T-Statistic": [stat],
        "P-Value": [p]
    })

    return results, p

# ---------- 5. Custom Analysis Placeholder ----------
def custom_analysis(df):
    # You can add your own custom analysis logic here
    return "Custom analysis results here."

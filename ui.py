import streamlit as st
import pandas as pd

# Page configuration
PAGE_CONFIG = {
    "page_title": "Logistic Regression Modeler",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Background styling CSS
PAGE_STYLING = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
}
[data-testid="stHeader"] {
    background: rgba(64, 60, 63, 0.88);
}
</style>
"""

def load_styling():
    """Injects the custom CSS for the app background."""
    st.markdown(PAGE_STYLING, unsafe_allow_html=True)

def create_sidebar(df):
    """Creates the sidebar for feature and target selection."""
    st.sidebar.header("Select Features and Target Variable")
    features = st.sidebar.multiselect("Features", df.columns.tolist())
    target = st.sidebar.selectbox("Target Variable", df.columns.tolist())
    return features, target

def create_prediction_form(features):
    """Creates the form for new data prediction."""
    st.subheader("Predict New Data")
    new_data = {}
    with st.form("prediction_form"):
        for feature in features:
            new_data[feature] = st.number_input(feature, value=0.0)
        
        # Submit button for the form
        submitted = st.form_submit_button("Predict")
        
    if submitted:
        return pd.DataFrame(new_data, index=[0])
    return None
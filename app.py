import streamlit as st
import pandas as pd
from st_aggrid import AgGrid

# Import your new modules
import ui
import model
import plotting

# Set page config
st.set_page_config(**ui.PAGE_CONFIG)

# Load custom styling
ui.load_styling()

# Title of the app
st.title("📊 Interactive Logistic Regression Modeler")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Choose a CSV file to get started", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
else:
    # --- 2. Data Loading & Sidebar ---
    try:
        df = pd.read_csv(uploaded_file)
        features, target = ui.create_sidebar(df)

        # --- 3. Data Explorer ---
        st.header("Data Explorer")
        st.write("Use the grid to sort, filter, and explore your data.")
        AgGrid(df, height=300, fit_columns_on_grid_load=True)
        
        with st.expander("View Data Statistics"):
            st.write(df.describe())

        # --- 4. Main App Logic ---
        if features and target:
            
            # --- 5. Exploratory Data Analysis (EDA) ---
            st.header("Exploratory Data Analysis (EDA)")
            plot_col_1, plot_col_2 = st.columns(2)
            
            with plot_col_1:
                st.subheader("Feature Distributions")
                feature_to_plot = st.selectbox("Select feature to plot", features)
                if feature_to_plot:
                    fig1 = plotting.plot_feature_distribution(df, feature_to_plot, target)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with plot_col_2:
                st.subheader("Correlation Heatmap")
                fig2 = plotting.plot_correlation_heatmap(df, features)
                st.plotly_chart(fig2, use_container_width=True)

            # --- 6. Model Training ---
            st.header("Model Training & Evaluation")
            
            with st.spinner("Training model..."):
                model_results = model.train_model(df, features, target)
            
            st.success("Model trained successfully!")
            
            # --- 7. Model Evaluation ---
            st.metric("Model Accuracy", f"{model_results['accuracy']:.2%}")
            
            eval_col_1, eval_col_2 = st.columns(2)
            
            with eval_col_1:
                st.subheader("Confusion Matrix")
                # Get unique classes for labels
                labels = sorted(df[target].unique())
                fig3 = plotting.plot_confusion_matrix(model_results['confusion_matrix'], labels=labels)
                st.plotly_chart(fig3, use_container_width=True)
            
            with eval_col_2:
                st.subheader("ROC Curve")
                fpr, tpr, roc_auc = model_results['roc_curve']
                fig4 = plotting.plot_roc_curve(fpr, tpr, roc_auc)
                st.plotly_chart(fig4, use_container_width=True)

            # --- 8. Prediction Form ---
            st.header("Make New Predictions")
            new_data = ui.create_prediction_form(features)
            
            if new_data is not None:
                prediction = model.get_prediction(
                    model_results['model'], 
                    model_results['scaler'], 
                    new_data
                )
                st.success(f"The predicted value for the target is: **{prediction[0]}**")
        
        else:
            st.info("Please select your features and a target variable from the sidebar.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
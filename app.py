# --- V4 --- THIS IS A TEST TO FORCE A REBUILD ---
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

# Import your new modules
import ui
import model
import plotting

# --- 1. Page Config & Data Load ---
st.set_page_config(**ui.PAGE_CONFIG)
ui.load_styling()

# Define the path to your default dataset
# (This assumes 'kindey_stone_urine_analysis.csv' is in the same folder as app.py)
DEFAULT_DATA_PATH = "kindey_stone_urine_analysis.csv" 

@st.cache_data  # Cache the data loading
def load_data(path):
    """Loads data from a specified path, with error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Dataset '{path}' not found. Please upload a file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading {path}: {e}")
        return None

# --- 2. Title & File Uploader ---
st.title("📊 Interactive Modeler (V4)") # <-- VISUAL MARKER
uploaded_file = st.file_uploader(
    "Upload a new CSV file to override the default dataset", type="csv"
)

df = None  # Initialize dataframe
if uploaded_file is not None:
    # A new file has been uploaded
    st.success("New data uploaded. Using this file for analysis.")
    df = load_data(uploaded_file)
else:
    # No file uploaded, load the default
    df = load_data(DEFAULT_DATA_PATH)
    if df is not None:
        st.info("Displaying default dataset. Upload a new CSV to analyze your own data.")

# --- 3. Main App Logic ---
if df is None:
    st.warning("No data to display. Please upload a CSV file.")
else:
    # --- 4. Sidebar (moved from original position) ---
    features, target = ui.create_sidebar(df)

    # --- 5. Data Explorer ---
    st.header("Data Explorer")
    st.write("Use the grid to sort, filter, and explore your data.")
    
    # --- FIX for AgGrid warning ---
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_grid_options(domLayout='autoHeight') # Use domLayout
    gridOptions = gb.build()
    
    AgGrid(df, 
           gridOptions=gridOptions,
           height=300, 
           allow_unsafe_jscode=True, # Set it to True to allow jsfunction to be injected
           )
    
    with st.expander("View Data Statistics"):
        st.write(df.describe())

    # --- 6. Main App Logic ---
    if features and target:
        
        # --- 7. Exploratory Data Analysis (EDA) ---
        st.header("Exploratory Data Analysis (EDA)")
        plot_col_1, plot_col_2 = st.columns(2)
        
        with plot_col_1:
            st.subheader("Feature Distributions")
            feature_to_plot = st.selectbox("Select feature to plot", features)
            if feature_to_plot:
                try:
                    fig1 = plotting.plot_feature_distribution(df, feature_to_plot, target)
                    st.plotly_chart(fig1, width='stretch') # --- FIX for deprecation ---
                except Exception as e:
                    st.warning(f"Could not plot distribution: {e}")
        
        with plot_col_2:
            st.subheader("Correlation Heatmap")
            try:
                fig2 = plotting.plot_correlation_heatmap(df, features)
                st.plotly_chart(fig2, width='stretch') # --- FIX for deprecation ---
            except Exception as e:
                st.warning(f"Could not plot heatmap: {e}")

        # --- 8. Model Training ---
        st.header("Model Training & Evaluation")
        
        # --- START OF THE CRITICAL FIX ---
        # We initialize model_results as an error.
        # This GUARANTEES that 'status' will exist.
        model_results = {
            "status": "error",
            "message": "Model training failed. Please check your data and feature selection (e.g., ensure target is not continuous)."
        }
        # --- END OF THE CRITICAL FIX ---
        
        try:
            with st.spinner("Training model..."):
                # This will *overwrite* the error dictionary if training is successful
                model_results = model.train_model(df, features, target)
        except Exception as e:
            # If a totally unexpected error happens, we catch it.
            # 'model_results' will keep its default error state.
            st.error(f"An unexpected error occurred during model training: {e}")
            pass
            
        # --- Check if model training was successful ---
        # This line CANNOT fail now, because model_results['status'] is guaranteed to exist.
        if model_results['status'] == 'error':
            st.error(model_results['message'])
        else:
            st.success("Model trained successfully!")
            
            # --- 9. Model Evaluation ---
            st.metric("Model Accuracy", f"{model_results['accuracy']:.2%}")
            
            # Check if multiclass warning is needed
            if not model_results['is_binary']:
                st.info("Multiclass classification detected. ROC curve is disabled.")
            
            eval_col_1, eval_col_2 = st.columns(2)
            
            with eval_col_1:
                st.subheader("Confusion Matrix")
                # Get unique classes for labels from model results
                labels = model_results['labels']
                fig3 = plotting.plot_confusion_matrix(model_results['confusion_matrix'], labels=labels)
                st.plotly_chart(fig3, width='stretch') # --- FIX for deprecation ---
            
            with eval_col_2:
                # Only show ROC curve if it's a binary problem
                if model_results['is_binary']:
                    st.subheader("ROC Curve")
                    fpr, tpr, roc_auc = model_results['roc_curve']
                    fig4 = plotting.plot_roc_curve(fpr, tpr, roc_auc)
                    st.plotly_chart(fig4, width='stretch') # --- FIX for deprecation ---
                else:
                    st.subheader("ROC Curve (Disabled)")
                    st.write("ROC curves are only generated for binary classification tasks.")
            
            # --- 10. Prediction Form ---
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
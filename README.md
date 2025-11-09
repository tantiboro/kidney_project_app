📊 Interactive Logistic Regression Modeler

An interactive web application built with Streamlit that allows users to upload a CSV dataset, train a Logistic Regression model, and visualize the model's performance—all in one place.

This app is designed to be a flexible tool for basic data exploration and classification modeling, handling both binary and multiclass target variables.

✨ Features

File Upload: Upload any CSV file to begin.

Data Exploration: View, sort, and filter the entire dataset using an interactive AgGrid.

Dynamic EDA:

Feature Distribution: Analyze how feature values are distributed across different target classes.

Correlation Heatmap: Automatically generate a heatmap to check for multicollinearity in your selected features.

Model Training:

Select any combination of features and a target variable from your data.

Automatically trains a Logistic Regression model.

Smart Target Handling: Intelligently detects if the target is binary or multiclass and adjusts the model (using One-vs-Rest) and evaluations accordingly.

Model Evaluation:

View key metrics like Model Accuracy.

See a detailed Confusion Matrix for all classes.

ROC Curve: For binary classification tasks, an ROC curve and AUC score are automatically generated.

Live Predictions: Use a simple form to input new data and get an instant prediction from the trained model.

⚙️ Project Structure

This app was refactored for a clean, modular structure, making it easy to maintain and test. This is ideal for CI/CD pipelines.

app.py: The main application file that coordinates the UI and logic.

ui.py: Contains all Streamlit UI components (sidebar, forms, styling).

model.py: Handles all data processing, model training, and prediction logic using scikit-learn.

plotting.py: Generates all data visualizations using Plotly.

requirements.txt: Lists all dependencies for easy setup.

🚀 How to Run Locally

Clone the repository:

git clone <your-repo-url>
cd <your-repo-directory>


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


🛠️ Tech Stack

Python 3.10+

Streamlit: For the interactive web app framework.

Pandas: For data manipulation.

Scikit-learn: For data scaling and Logistic Regression modeling.

Plotly: For interactive data visualizations.

streamlit-aggrid: For the interactive data table.
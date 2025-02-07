import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set Streamlit page configuration
st.set_page_config(page_title="Risk Analytics Platform", layout="wide")

# Custom Branding
st.sidebar.image("https://via.placeholder.com/150", caption="Your Company Logo", use_column_width=True)
st.sidebar.title("Risk Analytics Platform")
st.sidebar.markdown("Empowering banks with AI-driven risk insights.")

# Main Title
st.title("📊 AI-Powered Risk Analytics Platform")
st.write("Upload your dataset and analyze different financial risks including fraud, credit, ESG, and more.")

# File upload
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview:")
    st.write(df.head())
    
    # Select target variable
    target_variable = st.selectbox("🎯 Select Dependent Variable (Target)", df.columns)
    
    # Select independent variables
    feature_variables = st.multiselect("📊 Select Independent Variables", df.columns.drop(target_variable))
    
    if feature_variables:
        # Data visualization
        st.write("### 📈 Data Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Model selection and training
        X = df[feature_variables]
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_choice = st.selectbox("🤖 Choose ML Model", ["Logistic Regression", "Random Forest", "XGBoost"])
        
        if st.button("🚀 Run Model"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = XGBClassifier()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### ✅ Model Accuracy: {accuracy:.2f}")
            
            st.write("### 📊 Classification Report")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion matrix
            st.write("### 🔵 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

# Deployment Tip: To deploy on Streamlit Cloud, save this script as `app.py` and push it to GitHub, then connect it to Streamlit Cloud.

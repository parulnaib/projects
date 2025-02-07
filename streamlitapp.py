import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Risk Classification Dashboard", layout="wide")
st.title("Customer Risk Classification Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload Customer Data CSV", type=['csv'])
st.write("Upload a CSV file containing customer data such as age, income, transaction history, etc.")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # ML Settings
    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox("Select Target Variable", df.columns)

    with col2:
        features = st.multiselect("Select Features", 
                                [col for col in df.columns if col != target],
                                default=[col for col in df.columns if col != target])

    # Algorithm selection
    algorithm = st.selectbox("Select Algorithm", 
                           ["Random Forest", "SVM", "XGBoost", "CNN"])

    problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

    if st.button("Train Model"):
        # Prepare data
        X = df[features]
        y = df[target]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                           test_size=0.2, 
                                                           random_state=42)

        # Model training
        if algorithm == "Random Forest":
            if problem_type == "Classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                st.write("""
                ### Customer Risk Categories:
                - Low Risk (0): Reliable customers with stable profiles
                - Medium Risk (1): Customers requiring moderate monitoring
                - High Risk (2): Customers requiring enhanced due diligence
                """)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

        elif algorithm == "SVM":
            if problem_type == "Classification":
                model = SVC(kernel='rbf')
            else:
                model = SVR(kernel='rbf')

        elif algorithm == "XGBoost":
            if problem_type == "Classification":
                model = XGBClassifier()
            else:
                model = XGBRegressor()

        elif algorithm == "CNN":
            # Reshape data for CNN
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential([
                Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
                MaxPooling1D(2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid' if problem_type == "Classification" else 'linear')
            ])
            model.compile(optimizer='adam',
                        loss='binary_crossentropy' if problem_type == "Classification" else 'mse',
                        metrics=['accuracy'] if problem_type == "Classification" else ['mae'])

            # Train CNN
            history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)
            score = model.evaluate(X_test_cnn, y_test)
            st.write(f"Test Score: {score}")
            return

        # Train model (for non-CNN algorithms)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Display results
        st.success(f"Model Training Complete!")
        st.write(f"Model Score: {score:.4f}")

        # Feature importance for Random Forest and XGBoost
        if algorithm in ["Random Forest", "XGBoost"]:
            importances = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig = go.Figure(go.Bar(
                x=importances['feature'],
                y=importances['importance']
            ))
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance"
            )
            st.plotly_chart(fig)

            # Model Insights
            st.subheader("Model Insights")
            
            # Performance insights
            st.write("### Performance Analysis")
            if score > 0.9:
                st.success(f"Strong model performance with score of {score:.2f}")
            elif score > 0.7:
                st.info(f"Moderate model performance with score of {score:.2f}")
            else:
                st.warning(f"Model performance needs improvement with score of {score:.2f}")

            # Feature insights
            if algorithm in ["Random Forest", "XGBoost"]:
                st.write("### Feature Insights")
                top_features = importances.head(3)['feature'].tolist()
                st.write(f"Top influential features: {', '.join(top_features)}")
                
                # Recommendations
                st.write("### Recommendations")
                st.write("Based on the analysis:")
                st.write("1. Focus on collecting more data for top features")
                st.write("2. Consider feature engineering to improve model performance")
                if score < 0.8:
                    st.write("3. Try hyperparameter tuning to improve model accuracy")

else:
    st.info("Please upload a CSV file to begin analysis.")

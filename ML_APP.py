import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore') 
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Automated ML App")
st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.sidebar.success("Data uploaded successfully!")
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

st.sidebar.header("Data Exploration")

st.subheader("Data Shape")
st.write("Number of Rows:", df.shape[0])
st.write("Number of Columns:", df.shape[1])

st.subheader("Data Types")
st.write(df.dtypes)

numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns

label_encoder = LabelEncoder()

for column in categorical_columns:
    if column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])

columns_with_special_chars = df.columns[df.applymap(lambda x: isinstance(x, str) and not x.isnumeric() and not x.isalnum()).any()]

for column in df.columns:
    if df[column].dtype == 'object':
        mode_value = df[column].mode().iloc[0]
        df[column] = df[column].replace(regex=r'[^a-zA-Z0-9]', value=mode_value)
    else:
        df[column] = df[column].replace(regex=r'[^0-9.]', value=np.nan)
        df[column] = df[column].astype(float)
        mean_value = df[column].mean()
        df[column] = df[column].fillna(mean_value)

st.subheader("Summary Statistics")
st.write(df.describe())

st.sidebar.header("Data Visualization")

numeric_df = df.select_dtypes(include=['number'])

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

numeric_df = df.select_dtypes(include=['number'])

st.subheader("Attribute Distributions")
for column in numeric_df.columns:
    warnings.filterwarnings('ignore') 
    st.write(f"**{column}**")
    plt.figure(figsize=(6, 4))
    plt.hist(numeric_df[column], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    st.pyplot()

st.sidebar.header("Data Preprocessing and Model Building")

target_column = st.sidebar.text_input("Enter Target Column Name")

st.sidebar.header("Outlier Handling")

outlier_handling_method = st.sidebar.selectbox("Select Outlier Handling Method", ["None", "Remove Outliers", "Transform Outliers"])
threshold = 1.5

if outlier_handling_method != "None":
    st.write(f"Outlier Handling Method: {outlier_handling_method}")
    if outlier_handling_method == "Remove Outliers":
        Q1 = df[numeric_columns].quantile(0.25)
        Q3 = df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df[~((df[numeric_columns] < (Q1 - threshold * IQR)) | (df[numeric_columns] > (Q3 + threshold * IQR))).any(axis=1)]
        st.write("Number of Outliers Removed:", len(df) - len(df_no_outliers))
        df = df_no_outliers
    elif outlier_handling_method == "Transform Outliers":
        for column in numeric_columns:
            if column != target_column:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[column] = np.where(df[column] < lower_bound, lower_bound, np.where(df[column] > upper_bound, upper_bound, df[column]))

st.dataframe(df)

best_classifier = 0
best_regressor = 0

if target_column in df.columns:
    if len(df[target_column].unique()) <= 5:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    if problem_type == "Classification":
        model = RandomForestClassifier()
    elif problem_type == "Regression":
        model = RandomForestRegressor()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    model.fit(X, y)

    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.subheader("Feature Importance")
    st.write(feature_importance_df)

    upper_threshold = 0.05
    lower_threshold = -0.05

    positive_features = feature_importance_df[feature_importance_df['Importance'] >= upper_threshold]['Feature'].tolist()
    negative_features = feature_importance_df[feature_importance_df['Importance'] <= lower_threshold]['Feature'].tolist()

    df = df[positive_features + negative_features + [target_column]]
    if problem_type == "Classification":
        st.write("Selected Problem Type: Classification")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifiers = [
            ("Random Forest", RandomForestClassifier()),
            ("Logistic Regression", LogisticRegression()),
            ("Gradient Boosting", GradientBoostingClassifier())
        ]

        best_classifier = None
        best_accuracy = 0

        st.subheader("Model Evaluation for Classification")

        for name, classifier in classifiers:
            st.write(f"**{name}**")
            
            param_grid = {}
            grid_search = GridSearchCV(classifier, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.write(f"Accuracy: {accuracy:.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = name

        st.write(f"Best Classifier: {best_classifier} (Accuracy: {best_accuracy:.2f})")

    elif problem_type == "Regression":
        st.write("Selected Problem Type: Regression")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regressors = [
            ("Random Forest", RandomForestRegressor()),
            ("Linear Regression", LinearRegression()),
            ("Gradient Boosting", GradientBoostingRegressor())
        ]

        best_regressor = None
        best_rmse = float("inf")

        st.subheader("Model Evaluation for Regression")

        for name, regressor in regressors:
            st.write(f"**{name}**")
            
            param_grid = {}
            grid_search = GridSearchCV(regressor, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"R-squared (R2): {r2:.2f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_regressor = name

        st.write(f"Best Regressor: {best_regressor} (RMSE: {best_rmse:.2f})")

    if best_regressor:
        
        st.subheader("Regression Plot of Actual vs. Predicted Values")
        plt.figure(figsize=(8, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha":0.5})
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        st.pyplot()
        download_button = st.button("Download Best Model")

    elif best_classifier:
        
        st.subheader("Confusion Matrix")
        confusion = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=grid_search.classes_)
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        
        st.pyplot()
        download_button = st.button("Download Best Model")

    if download_button:
        best_model = grid_search
        with open("best_model.pkl", "wb") as model_file:
            pickle.dump(best_model, model_file)
        
        st.success("Best model has been downloaded as 'best_model.pkl'.")
else:
    st.sidebar.warning("Please enter a valid target column name.")

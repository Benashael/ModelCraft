import streamlit as st
import pandas as pd
import numpy as np
# from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, f1_score, mean_absolute_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import base64
from scipy.stats import zscore 
import shap 
import matplotlib
import IPython
import plotly
import plotly.graph_objs as go

st.set_page_config(page_title="Model Craft", page_icon="🤖", layout="wide")

# App description
st.title("✨🚀 Model Craft: Accelerate Model Building and Optimization")

# Define a function to load your dataset
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Upload a dataset
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

# Initialize data as None
data = None

# Check if a file was uploaded and if it's valid
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
    except (ValueError, pd.errors.ParserError):
        st.error("❌ The uploaded dataset is not in a valid format or language. Please upload a valid dataset in CSV format.")
        data = None  # Set data to None if it's not valid

# Create Streamlit pages
page = st.sidebar.radio("**Select a Page 📄**", [
    "🏠 Home Page", 
    "📋 Data Profiling", 
    "🔑 Data Encoding", 
    "🛠️ Data Preprocessing", 
    "🧹 Data Cleaning", 
    "📊 Data Visualization", 
    "🎯 Feature Selection", 
    "⚙️ Hyperparameter Tuning", 
    "🤖 ML Model Selection", 
    "📂 Classification (ML)", 
    "📉 Regression (ML)", 
    "🔍 Clustering (ML)", 
    "📈 Model Evaluation"
])

# Introduction Page
if page == "🏠 Home Page":

    st.header("👋 Welcome to the Model Craft Application!")
    st.write("This application is designed to help you streamline the process of data analysis and machine learning model selection. Follow the steps below to make the most of this application:")

    st.subheader("🏠 Home Page")
    st.markdown(
        "The **Home Page** is the starting point of the application. You can navigate to different sections of the app using the sidebar navigation."
    )

    st.subheader("📋 Data Profiling Page")
    st.markdown(
        "The **Data Profiling Page** allows you to gain a detailed understanding of your dataset. Explore dataset shape, column names, data types, summary statistics, categorical features, missing values, correlation matrix, and data head."
    )

    st.subheader("🔑 Data Encoding Page")
    st.markdown(
        "The **Data Encoding Page** empowers you to encode categorical variables in the dataset. Select a dataset and apply different encoding techniques to handle categorical variables."
    )

    st.subheader("🛠️ Data Preprocessing Page")
    st.markdown(
        "The **Data Preprocessing Page** prepares your dataset for machine learning by scaling features, splitting data, and handling outliers."
    )
    
    st.subheader("🧹 Data Cleaning Page")
    st.markdown(
        "The **Data Cleaning Page** allows you to clean missing values in your dataset. Select a dataset and apply different cleaning techniques to handle missing values."
    )

    st.subheader("📊 Data Visualization Page")
    st.markdown(
        "The **Data Visualization Page** lets you visualize the dataset using various techniques such as histograms, scatter plots, and heat maps. Select a dataset and explore its visual representations."
    )

    st.subheader("🎯 Feature Selection Page")
    st.markdown(
        "The **Feature Selection Page** allows you to choose and analyze the most important features in your dataset for machine learning. Feature selection is a critical step to improve model performance and reduce the complexity of your model."
    )
    
    st.subheader("⚙️ Hyperparameter Tuning Page")
    st.markdown(
        "The **Hyperparameter Tuning Page** allows you to fine-tune machine learning models with user-selected hyperparameters for optimal performance."
    )

    st.subheader("🤖 ML Model Selection Page")
    st.markdown(
        "The **ML Model Selection Page** helps you choose the right machine learning model based on the problem type (classification, regression, or time series). Pick a dataset and select the target variable to find the best machine learning model."
    )

    st.subheader("📂 Classification (ML) Page")
    st.markdown(
        "The **Classification (ML) Page** allows you to perform automated machine learning (AutoML) for classification problems using the `lazyClassifier` library. Select a dataset, choose the target variable, and run the AutoML algorithm."
    )

    st.subheader("📉 Regression (ML) Page")
    st.markdown(
        "The **Regression (ML) Page** enables you to perform automated machine learning (AutoML) for regression problems using the `lazyRegression` library. Select a dataset, choose the target variable, and run the AutoML algorithm."
    )

    st.subheader("🔍 Clustering (ML) Page")
    st.markdown(
        "The **Clustering (ML) Page** empowers you to perform automated machine learning (AutoML) for clustering problems. Clustering is a technique used to group similar data points together. In this page, you can select a dataset and choose the number of clusters. Note: This task is suitable for small datasets."
    )

    st.subheader("📈 Model Evaluation Page")
    st.markdown(
        "The **Model Evaluation Page** is where you can evaluate machine learning models on your dataset. Choose the problem type (classification or regression), select X and Y variables, and pick a model to see evaluation results."
    )
    
    st.subheader("📝 Using the App")
    st.markdown(
        "1. Start on the **Home Page**, and then navigate to the pages that match your needs."
    )
    st.markdown(
        "2. Ensure you upload a dataset in the **Data Cleaning Page** before proceeding to other pages that require a dataset."
    )
    st.markdown(
        "3. Use the sidebar navigation to switch between pages and follow the instructions on each page to complete the tasks."
    )
    st.markdown(
        "4. Make sure to select the appropriate problem type (classification or regression) and follow any additional instructions for model selection and evaluation."
    )

    st.subheader("💡 Additional Tips")
    st.markdown(
        "5. You can always refer back to the **Introduction** page for a quick overview of the app's functionality and how to use it."
    )
    st.markdown(
        "6. Don't hesitate to reach out for assistance or if you have any questions about using the application effectively."
    )
    st.markdown(
        "7. You can also download the preprocessed dataset, cleaned dataset, encoded dataset, training/testing datasets and clustered dataset for further analysis."
    )

    st.markdown(
        "🎉 Enjoy using the Model Craft Application and have a productive data analysis and model selection experience!"
    )

# Data Profiling Page
elif page == "📋 Data Profiling":
    st.header("📋 Data Profiling")

    if data is not None and not data.empty:
        st.write("### 🔍 Dataset Overview")

        # Dataset Shape
        st.write("**🗂️ Dataset Shape:**", data.shape)

        # Column Names
        st.subheader("📑 Column Names:")
        st.write(data.columns)

        # Data Types
        st.subheader("📂 Data Types:")
        st.write(data.dtypes)

        # Summary Statistics
        st.subheader("📊 Summary Statistics:")
        st.write(data.describe())

        # Categorical Features
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        st.subheader("🔠 Categorical Features:")
        if categorical_features:
            st.write("**Number of Categorical Features:**", len(categorical_features))
            st.write("**Categorical Feature Names:**")
            st.write(categorical_features)
        else:
            st.write("🚫 No Categorical Features in the dataset.")

        # Missing Values
        missing_values = data.isnull().sum()
        st.subheader("⚠️ Missing Values:")
        if missing_values.sum() > 0:
            st.write("**Total Missing Values:**", missing_values.sum())
            st.write("**Missing Values by Column:**")
            st.write(missing_values[missing_values > 0])
        else:
            st.write("🎉 No missing values in the dataset.")

        # Correlation Matrix (only if there are no categorical features)
        st.subheader("📈 Correlation Matrix:")
        if not categorical_features:
            correlation_matrix = data.corr()
            st.write(correlation_matrix)
        else:
            st.write("⚠️ Correlation matrix is not calculated because categorical features are present.")

        # Data Head
        st.subheader("🔝 Data Head (First 5 Rows):")
        st.write(data.head())

        # Data Tail
        st.subheader("🔚 Data Tail (Last 5 Rows):")
        st.write(data.tail()) 

    else:
        st.error("🚨 Please upload a valid dataset in the 'Data Profiling' step to continue.")

# Data Encoding Page
elif page == "🔑 Data Encoding":
    st.header("🔑 Data Encoding")
    if data is not None:
        st.write("### 🗂️ Dataset Overview:")
        st.write(data)
        st.write("**📏 Dataset Shape:**", data.shape)

        # Define the maximum allowed dataset size for data encoding
        max_rows_for_encoding = 5000
        max_columns_for_encoding = 50

        if data.shape[0] > max_rows_for_encoding or data.shape[1] > max_columns_for_encoding:
            st.warning(f"⚠️ Note: The dataset size exceeds the maximum allowed for data encoding (max rows: {max_rows_for_encoding}, max columns: {max_columns_for_encoding}).")
        else:
            st.subheader("🔠 Encode Categorical Variables")
            categorical_cols = data.select_dtypes(include=["object"]).columns

            if not categorical_cols.empty:
                st.write(f"**📝 Categorical Columns Available:** {', '.join(categorical_cols)}")
                selected_cols = st.multiselect("📋 Select Categorical Columns to Encode", categorical_cols)

                if not selected_cols:
                    st.warning("⚠️ Please select one or more categorical columns to encode.")
                else:
                    for col in selected_cols:
                        encoding_option = st.radio(
                            f"🎯 Select Encoding Method for '{col}'",
                            ["Label Encoding", "One-Hot Encoding"],
                            key=f"encoding_option_{col}"
                        )

                        if encoding_option == "Label Encoding":
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col])
                            st.success(f"✅ Applied Label Encoding on '{col}'.")
                        else:
                            data = pd.get_dummies(data, columns=[col], prefix=[col])
                            st.success(f"✅ Applied One-Hot Encoding on '{col}'.")

                    st.write("### ✅ Data After Encoding:")
                    st.write(data)

                    # Allow users to download the encoded dataset
                    st.subheader("📥 Download Encoded Dataset")
                    encoded_csv = data.to_csv(index=False).encode()
                    st.download_button(
                        label="💾 Download Encoded Dataset as CSV",
                        data=encoded_csv,
                        file_name="encoded_data.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("🚫 No categorical columns found in the dataset for encoding.")
    else:
        st.warning("🚨 Please upload a dataset to proceed with encoding.")

# Data Preprocessing Page
elif page == "🛠️ Data Preprocessing":
    st.header("🛠️ Data Preprocessing")

    # Check if the dataset is available
    if data is not None and not data.empty:
        st.write("### 🗂️ Dataset Overview:")
        st.write(data)
        st.write("**📏 Dataset Shape:**", data.shape)

        st.subheader("🔄 Data Preprocessing Steps")

        # Step 1: Feature Scaling and Normalization
        if data.shape[0] <= 5000 and data.shape[1] <= 50:
            st.subheader("Step 1: Feature Scaling and Normalization")

            numerical_cols = data.select_dtypes(include=[np.number]).columns

            if not numerical_cols.empty:
                selected_scaling = st.radio("📊 Select Feature Scaling or Normalization Method:", ["Min-Max Scaling", "Standardization"])

                if selected_scaling == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
                    st.success("✅ Applied Min-Max Scaling:")
                    st.write(data)

                    st.download_button(
                        label="💾 Download Scaled Data (Min-Max Scaling)",
                        data=data.to_csv(index=False).encode(),
                        file_name="scaled_data_minmax.csv",
                        mime="text/csv"
                    )

                elif selected_scaling == "Standardization":
                    scaler = StandardScaler()
                    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
                    st.success("✅ Applied Standardization:")
                    st.write(data)

                    st.download_button(
                        label="💾 Download Standardized Data",
                        data=data.to_csv(index=False).encode(),
                        file_name="standardized_data.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ No numerical columns found for scaling or normalization.")
        else:
            st.warning("⚠️ Feature Scaling and Normalization is limited to datasets with a maximum of 5000 rows and 50 columns.")

        # Step 2: Data Splitting
        st.subheader("Step 2: Data Splitting (Train-Test Split)")
        test_size = st.slider("📊 Select Test Data Proportion:", 0.1, 0.5, step=0.05)

        target_variable = st.selectbox("🎯 Select Target Variable (y):", data.columns)
        other_variables = st.multiselect("📋 Select Other Variables (X):", [col for col in data.columns if col != target_variable])

        if target_variable and other_variables:
            X = data[other_variables]
            y = data[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.success(f"✅ Performed Train-Test Split (Test Size: {test_size:.2f})")

            st.download_button(
                label="💾 Download Training Data",
                data=pd.concat([X_train, y_train], axis=1).to_csv(index=False).encode(),
                file_name="training_data.csv",
                mime="text/csv"
            )

            st.download_button(
                label="💾 Download Testing Data",
                data=pd.concat([X_test, y_test], axis=1).to_csv(index=False).encode(),
                file_name="testing_data.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Please select a target variable and at least one other variable for the split.")

        # Step 3: Outlier Detection and Handling
        if data.shape[0] <= 5000 and data.shape[1] <= 50:
            st.subheader("Step 3: Outlier Detection and Handling")

            numerical_cols = data.select_dtypes(include=[np.number]).columns
            selected_outlier_method = st.radio("📊 Select Outlier Detection Method:", ["Z-Score", "IQR"])

            if not numerical_cols.empty:
                selected_column = st.selectbox("📋 Select Column for Outlier Detection:", numerical_cols)

                if selected_outlier_method == "Z-Score":
                    z_scores = zscore(data[selected_column])
                    outlier_indices = np.where(np.abs(z_scores) > 3)
                    data_no_outliers = data.drop(outlier_indices[0])
                    st.success("✅ Applied Z-Score Outlier Detection")
                    st.write(data_no_outliers)

                    st.download_button(
                        label="💾 Download Data after Z-Score Handling",
                        data=data_no_outliers.to_csv(index=False).encode(),
                        file_name="cleaned_data_zscore.csv",
                        mime="text/csv"
                    )

                elif selected_outlier_method == "IQR":
                    Q1 = data[selected_column].quantile(0.25)
                    Q3 = data[selected_column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_indices = ((data[selected_column] < lower_bound) | (data[selected_column] > upper_bound))
                    data_no_outliers = data[~outlier_indices]
                    st.success("✅ Applied IQR Outlier Detection")
                    st.write(data_no_outliers)

                    st.download_button(
                        label="💾 Download Data after IQR Handling",
                        data=data_no_outliers.to_csv(index=False).encode(),
                        file_name="cleaned_data_iqr.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ No numerical columns available for outlier detection.")
        else:
            st.warning("⚠️ Outlier Detection and Handling is limited to datasets with a maximum of 5000 rows and 50 columns.")

    else:
        st.warning("🚨 Please upload a dataset to proceed with preprocessing.")

# Data Cleaning Page
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning")

    # Check if the dataset is available
    if data is not None:
        st.write("📊 **Dataset:**")
        st.write(data)
        st.write("📏 **Dataset Shape:**")
        st.write(data.shape)

        # Option to Show Summary Statistics
        show_summary = st.checkbox("📈 Show Summary Statistics")
        if show_summary:
            st.subheader("📊 Summary Statistics")
            st.write(data.describe())

        # Check if the dataset has too many rows or columns
        max_rows_for_cleaning = 5000
        max_columns_for_cleaning = 50

        if data.shape[0] > max_rows_for_cleaning or data.shape[1] > max_columns_for_cleaning:
            st.warning(f"⚠️ **Note:** The dataset size exceeds the maximum allowed for data cleaning (max rows: {max_rows_for_cleaning}, max columns: {max_columns_for_cleaning}).")
        else:
            # Check if there are categorical features
            categorical_features = data.select_dtypes(include=['object']).columns.tolist()

            if not categorical_features:
                st.warning("ℹ️ **Note:** The dataset has no categorical features, so you can use the 'Mean' or 'Median' methods.")
            else:
                st.write("🗂️ **Categorical Features:**")
                st.write(categorical_features)

            # Choose missing value handling method
            st.subheader("🛠️ Missing Value Handling")
            methods = ["🚮 Drop Missing Values", "✏️ Custom Value"]
            if not categorical_features:
                methods.extend(["📊 Mean", "📉 Median"])

            method = st.selectbox("🔧 Select a method:", methods)

            if method == "🚮 Drop Missing Values":
                data_cleaned = data.dropna()
                st.write("✅ **Dropped missing values.**")
                st.write(data_cleaned)

            elif method == "✏️ Custom Value":
                custom_value = st.text_input("🔤 Enter a custom value to fill missing cells:", "0")
                data_cleaned = data.fillna(custom_value)
                st.write(f"✅ **Filled missing values with custom value:** {custom_value}")
                st.write(data_cleaned)

            elif (method == "📊 Mean" or method == "📉 Median") and not categorical_features:
                if method == "📊 Mean":
                    data_cleaned = data.fillna(data.mean())
                    st.write("✅ **Filled missing values with mean.**")
                    st.write(data_cleaned)
                else:  # method == "📉 Median"
                    data_cleaned = data.fillna(data.median())
                    st.write("✅ **Filled missing values with median.**")
                    st.write(data_cleaned)
            else:
                st.warning(f"⚠️ {method} method not available due to the presence of categorical features. Use '🚮 Drop Missing Values' or '✏️ Custom Value' instead.")

        # Allow users to download the cleaned dataset
        if st.button("💾 Download Cleaned Dataset"):
            cleaned_csv = data_cleaned.to_csv(index=False)
            cleaned_csv = cleaned_csv.encode()
            st.download_button(
                label="📥 Click here to download cleaned dataset as CSV",
                data=cleaned_csv,
                key="cleaned_data.csv",
                file_name="cleaned_data.csv"
            )
    else:
        st.warning("⚠️ Please upload a dataset in the 'Data Cleaning' step to continue.")

# Data Visualization Page
elif page == "📊 Data Visualization":
    st.header("📊 Data Visualization")

    if data is not None:
        # Filter out only numerical columns
        numerical_columns = data.select_dtypes(include=['int64', 'float64'])
        
        if not numerical_columns.empty:
            st.write("📊 **Dataset:**")
            st.write(data)

            st.subheader("📈 Data Visualization")
            st.write("🔍 **Select Columns for Visualization:**")
            columns_to_visualize = st.multiselect("📋 Select Columns", numerical_columns.columns)
            
            if columns_to_visualize:
                st.write("📉 **Line Chart:**")
                st.line_chart(data[columns_to_visualize])

                st.write("📊 **Bar Chart:**")
                st.bar_chart(data[columns_to_visualize])

                st.write("🌊 **Area Chart:**")
                st.area_chart(data[columns_to_visualize])
        else:
            st.warning("⚠️ No numerical columns found in the dataset. Please upload a dataset with numerical data to continue.")
    else:
        st.warning("⚠️ Please upload a dataset to continue.")
        
# Feature Selection Page
elif page == "🎯 Feature Selection":
    st.header("🎯 Feature Selection")

    # Check if the dataset is available and within size limits
    if data is not None and not data.empty:
        if data.shape[0] <= 5000 and data.shape[1] <= 50:
            st.subheader("📊 Dataset Overview")

            # Dataset Shape
            st.write("📏 **Dataset Shape:**", data.shape)

            # Check for categorical features
            categorical_features = data.select_dtypes(include=['object']).columns.tolist()

            if not categorical_features:
                # List of available features
                features = data.columns

                # Select Features for Analysis
                st.write("🔍 **Select Features for Analysis:**")
                selected_features = st.multiselect("📋 Choose Features", features)

                if len(selected_features) > 0:
                    # Display selected features
                    st.subheader("✅ Selected Features")
                    st.write(selected_features)

                    # Target Variable Selection
                    target_variable = st.selectbox("🎯 Select the Target Variable", features)

                    # Split data into X and y
                    X = data[selected_features]
                    y = data[target_variable]

                    # Feature Ranking
                    selector = SelectKBest(score_func=f_regression, k="all")
                    X_new = selector.fit_transform(X, y)

                    # Get feature scores and rankings
                    feature_scores = selector.scores_
                    feature_rankings = (-feature_scores).argsort().argsort()  # Rank features

                    # Display feature scores and rankings
                    feature_info = pd.DataFrame({
                        "Feature": selected_features,
                        "Score": feature_scores,
                        "Ranking": feature_rankings
                    })
                    st.subheader("📈 Feature Scores and Rankings")
                    st.write(feature_info.sort_values(by="Ranking"))

                    # Perform a simple regression using the top-ranked feature
                    top_feature = feature_info[feature_info["Ranking"] == 0]["Feature"].iloc[0]
                    X_top_feature = X[top_feature].values.reshape(-1, 1)
                    X_train, X_test, y_train, y_test = train_test_split(X_top_feature, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate Mean Squared Error as a metric
                    mse = mean_squared_error(y_test, y_pred)

                    st.subheader("📉 Regression using Top-Ranked Feature")
                    st.write(f"🏆 **Top-Ranked Feature:** {top_feature}")
                    st.write(f"📉 **Mean Squared Error (MSE):** {mse:.2f}")
                else:
                    st.warning("⚠️ Please select at least one feature for analysis.")
            else:
                st.error("🚫 The dataset contains categorical features and is not suitable for feature selection.")
        else:
            st.warning("⚠️ The dataset exceeds the size limits for this page (max rows: 5000, max columns: 50).")
    else:
        st.error("❌ Please upload a valid dataset in the 'Feature Selection' step to continue.")

# Hyperparameter Tuning Page
elif page == "⚙️ Hyperparameter Tuning":
    st.header("⚙️ Hyperparameter Tuning")

    # Check if the dataset and model are available
    if data is not None and not data.empty:
        st.subheader("📊 Dataset Overview")
        st.write("Dataset:")
        st.write(data)
        st.write("📏 **Dataset Shape:**", data.shape)

        # Check if the selected dataset has categorical features
        categorical_cols = data.select_dtypes(include=["object"]).columns
        if not categorical_cols.empty:
            st.error("🚫 Hyperparameter tuning is not supported for datasets with categorical features. Please preprocess your data first.")
        else:
            st.subheader("🤖 Select Machine Learning Model")
            selected_model = st.selectbox("🔍 Choose a Model", ["Logistic Regression (Classification)", "Ridge Regression (Regression)"])
            # Add more machine learning models as needed

            model = None
            hyperparameters = {}
            
            if selected_model == "Logistic Regression (Classification)":
                model = LogisticRegression()
                hyperparameters = {
                    "C": st.slider("📏 Inverse of Regularization Strength (C)", 0.001, 10.0),
                    "max_iter": st.slider("🔄 Maximum Iterations (max_iter)", 100, 1000, step=100),
                }

            elif selected_model == "Ridge Regression (Regression)":
                model = Ridge()
                hyperparameters = {
                    "alpha": st.slider("🛠️ Alpha (Regularization Strength)", 0.001, 10.0),
                    "fit_intercept": st.checkbox("✅ Fit Intercept", value=True),
                    "max_iter": st.slider("🔄 Maximum Number of Iterations", 100, 1000, step=100),
                }

            # Add hyperparameters for other models as needed

            if model is not None:
                st.subheader("🎯 Hyperparameter Tuning")

                # Display the selected hyperparameters
                st.write("⚙️ **Selected Hyperparameters:**")
                st.write(hyperparameters)

                try:
                    # Prompt the user to select target variable and other variables
                    st.subheader("📌 Select Target Variable and Other Variables")

                    target_variable = st.selectbox("🎯 Select the Target Variable (y)", data.columns)
                    other_variables = st.multiselect("📋 Select Other Variables (X)", [col for col in data.columns if col != target_variable])

                    if other_variables and target_variable:
                        X_train = data[other_variables]
                        y_train = data[target_variable]

                        # Perform hyperparameter tuning using RandomizedSearchCV
                        st.write("⏳ **Tuning Hyperparameters...**")
                        param_dist = {}
                        for param_name, param_value in hyperparameters.items():
                            if isinstance(param_value, float):
                                param_dist[param_name] = np.arange(param_value, param_value + 0.1, 0.01)
                            else:
                                param_dist[param_name] = list(range(param_value))

                        randomized_search = RandomizedSearchCV(
                            model, 
                            param_distributions=param_dist, 
                            n_iter=10, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                            n_jobs=-1
                        )
                        randomized_search.fit(X_train, y_train)

                        best_hyperparameters = randomized_search.best_params_

                        # Display the best hyperparameters and their performance
                        st.subheader("🏆 Best Hyperparameters")
                        st.write(best_hyperparameters)

                        # Display the model's performance with the best hyperparameters
                        st.subheader("📈 Model Performance with Best Hyperparameters")
                        best_model = randomized_search.best_estimator_

                        # Split the data for evaluation
                        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                        best_model.fit(X_train, y_train)
                        y_pred = best_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"✅ **Accuracy with Best Hyperparameters:** {accuracy:.2f}")

                    else:
                        st.error("⚠️ Please select a valid target variable and at least one other variable.")
                except Exception as e:
                    st.error(f"❌ An error occurred during hyperparameter tuning: {str(e)}")
            else:
                st.error("🚫 An error occurred while selecting the model. Please try again.")
    else:
        st.warning("⚠️ Please upload a dataset in the 'Data Cleaning' step to continue.")

# ML Model Selection Page
elif page == "🤖 ML Model Selection":
    st.header("🤖 ML Model Selection")

    # Function to check for categorical columns
    def has_categorical_columns(data):
        # Identifies if any column in the dataset is of type 'object' (categorical)
        return data.select_dtypes(include=['object']).empty

    # Check if a dataset is uploaded
    if data is not None:
        # Check for categorical columns
        if has_categorical_columns(data):
            st.subheader("📊 Dataset Overview")
            st.write("Dataset Preview:")
            st.write(data)
            st.write("📏 Dataset Shape:", data.shape)
    
            # Define dataset size constraints for ML model selection
            max_rows = 5000
            max_columns = 50
    
            if data.shape[0] > max_rows or data.shape[1] > max_columns:
                st.warning(f"⚠️ The dataset exceeds the allowed size for ML model selection (Max Rows: {max_rows}, Max Columns: {max_columns}).")
            else:
                st.subheader("🔍 Select Problem Type")
                problem_type = st.radio("📌 Problem Type", ["Classification", "Regression"])
    
                # Prompt to select the target variable
                target_variable = st.selectbox("🎯 Select Target Variable (y)", data.columns)
    
                # Identify feature columns
                feature_columns = [col for col in data.columns if col != target_variable]
                selected_features = st.multiselect("📋 Select Features (X)", feature_columns)
    
                if not selected_features:
                    st.warning("⚠️ Please select one or more features (X) to proceed.")
                else:
                    # Split data into features (X) and target (y)
                    X = data[selected_features]
                    y = data[target_variable]
    
                    # Split data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
                    if problem_type == "Classification":
                        st.subheader("🤖 Classification Models")
                        classification_models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]
                        for model in classification_models:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write(f"**Model:** {type(model).__name__}")
                            st.write(f"✅ **Accuracy Score:** {accuracy:.2f}")
    
                    elif problem_type == "Regression":
                        st.subheader("📈 Regression Models")
                        regression_models = [RandomForestRegressor(), LinearRegression(), Ridge(), Lasso()]
                        for model in regression_models:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"**Model:** {type(model).__name__}")
                            st.write(f"📉 **Mean Squared Error:** {mse:.2f}")
        else:
            st.error("🚫 The page does not support datasets with categorical columns. Please preprocess your data.")
    else:
        st.warning("📂 Please upload a dataset to continue.")

# AutoML for Classification Page
elif page == "📂 Classification (ML)":
    st.header("📂 Classification (ML)")

    if data is not None:
        st.subheader("📊 Dataset Overview")
        st.write("Dataset Preview:")
        st.write(data)
        st.write("📏 Dataset Shape:", data.shape)

        # Check if the dataset contains categorical columns
        has_categorical_columns = data.select_dtypes(include=['object']).empty

        if has_categorical_columns:
            st.subheader("🤖 AutoML for Classification")
            
            # Define maximum dataset size constraints
            max_rows = 5000
            max_columns = 50

            if data.shape[0] > max_rows or data.shape[1] > max_columns:
                st.error(f"⚠️ Dataset size exceeds the allowed limit for classification (Max Rows: {max_rows}, Max Columns: {max_columns}).")
            else:
                # Target variable selection
                target_variable = st.selectbox("🎯 Select Target Variable (Y)", data.columns)

                # Feature selection
                X_variables = st.multiselect("📋 Select Features (X)", [col for col in data.columns if col != target_variable])

                if not X_variables:
                    st.warning("⚠️ Please select one or more features (X) to proceed.")
                else:
                    # Test size and random state sliders
                    test_size = st.slider("📐 Select Test Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
                    random_state = st.slider("🎲 Select Random State", min_value=1, max_value=100, value=42, step=1)

                    # Splitting the dataset
                    X = data[X_variables]
                    Y = data[target_variable]
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

                    st.subheader("📈 Classification Models and Results")
                    classification_models = [
                        RandomForestClassifier(),
                        LogisticRegression(max_iter=1000),  # Increased iterations for better convergence
                        DecisionTreeClassifier()
                    ]

                    # Loop through each model and display results
                    for model in classification_models:
                        model.fit(X_train, Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = accuracy_score(Y_test, Y_pred)

                        st.write(f"**Model:** {type(model).__name__}")
                        st.write(f"✅ **Accuracy Score:** {accuracy:.2f}")
        else:
            st.warning("🚫 This page is only available for datasets without categorical columns. Please preprocess your data.")
    else:
        st.warning("📂 Please upload a dataset to continue.")

# AutoML for Regression Page 
elif page == "📉 Regression (ML)":
    st.header("📉 Regression (ML)")

    if data is not None:
        st.subheader("📊 Dataset Overview")
        st.write("Dataset Preview:")
        st.write(data)
        st.write("📏 Dataset Shape:", data.shape)

        # Check if the dataset contains categorical columns
        has_categorical_columns = data.select_dtypes(include=['object']).shape[1] > 0

        if has_categorical_columns:
            st.error("🚫 AutoML for Regression is enabled only for datasets without categorical columns. Please preprocess your data.")
        else:
            st.subheader("🤖 AutoML for Regression")

            # Define the maximum allowed dataset size for regression
            max_rows = 5000
            max_columns = 50

            if data.shape[0] > max_rows or data.shape[1] > max_columns:
                st.error(f"⚠️ Dataset size exceeds the maximum allowed for regression (Max Rows: {max_rows}, Max Columns: {max_columns}).")
            else:
                # Target variable selection
                target_variable = st.selectbox("🎯 Select Target Variable (Y)", data.columns)

                # Feature selection
                X_variables = st.multiselect("📋 Select Features (X)", [col for col in data.columns if col != target_variable])

                if not X_variables:
                    st.warning("⚠️ Please select one or more features (X) to proceed.")
                else:
                    # Test size and random state sliders
                    test_size = st.slider("📐 Select Test Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
                    random_state = st.slider("🎲 Select Random State", min_value=1, max_value=100, value=42, step=1)

                    # Splitting the dataset
                    X = data[X_variables]
                    Y = data[target_variable]
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

                    st.subheader("📈 Regression Models and Results")
                    regression_models = [
                        RandomForestRegressor(),
                        LinearRegression(),
                        Ridge(),
                        Lasso()
                    ]

                    # Loop through each model and display results
                    for model in regression_models:
                        model.fit(X_train, Y_train)
                        Y_pred = model.predict(X_test)
                        mse = mean_squared_error(Y_test, Y_pred)

                        st.write(f"**Model:** {type(model).__name__}")
                        st.write(f"📉 **Mean Squared Error:** {mse:.2f}")

                        # Display additional regression metrics if needed
                        st.write(f"📏 **R2 Score:** {model.score(X_test, Y_test):.2f}")
                
    else:
        st.warning("📂 Please upload a dataset to continue.")
        
# AutoML for Clustering Page
elif page == "🔍 Clustering (ML)":
    st.header("🔍 Clustering (ML)")

    # Check if the dataset is available
    if data is not None:
        st.subheader("📊 Dataset Overview")
        st.write("Dataset Preview:")
        st.write(data)
        st.write("📏 Dataset Shape:", data.shape)

        # Define the maximum allowed dataset size for clustering
        max_rows_for_clustering = 2000
        max_columns_for_clustering = 25

        if data.shape[0] > max_rows_for_clustering or data.shape[1] > max_columns_for_clustering:
            st.error(f"⚠️ Dataset size exceeds the maximum allowed for clustering (max rows: {max_rows_for_clustering}, max columns: {max_columns_for_clustering}).")
        else:
            # Select the number of clusters
            num_clusters = st.slider("🧩 Select the Number of Clusters:", 2, 10)

            # Remove rows with missing values
            X = data.dropna()

            if X.shape[0] < num_clusters:
                st.error("⚠️ Not enough data points for the selected number of clusters.")
            else:
                # One-hot encoding for categorical features
                categorical_features = X.select_dtypes(include=['object']).columns.tolist()
                if categorical_features:
                    st.write(f"🎯 Categorical features detected: {', '.join(categorical_features)}")
                    X_encoded = pd.get_dummies(X, columns=categorical_features)
                else:
                    X_encoded = X

                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                X_encoded['Cluster'] = kmeans.fit_predict(X_encoded)

                # Display the clustering result
                st.write(f"🔍 Performed K-Means clustering with {num_clusters} clusters.")
                st.write("📊 Clustered Data Preview:")
                st.write(X_encoded)

                # Provide download link for the dataset with clusters
                csv_data_encoded = X_encoded.to_csv(index=False)
                b64 = base64.b64encode(csv_data_encoded.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="clustered_data.csv">📥 Download Clustered Dataset</a>'
                st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("📂 Please upload a dataset in the 'AutoML for Clustering' step to continue.")

# Model Evaluation Page
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation")

    # Define the maximum allowed dataset size for model evaluation
    max_rows_for_evaluation = 5000
    max_columns_for_evaluation = 50

    if data is not None:
        # Check for categorical columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            st.warning("⚠️ The dataset contains categorical columns. Model evaluation is only supported for datasets without categorical columns.")
        elif data.shape[0] > max_rows_for_evaluation or data.shape[1] > max_columns_for_evaluation:
            st.warning(f"⚠️ Note: The dataset size exceeds the maximum allowed for model evaluation (max rows: {max_rows_for_evaluation}, max columns: {max_columns_for_evaluation}).")
        else:
            # Select Problem Type
            problem_type = st.radio("🔧 Select Problem Type", ["Classification", "Regression"])

            if problem_type == "Classification":
                st.subheader("📊 Classification Model Evaluation")

                classification_models = ["Random Forest Classifier", "Logistic Regression", "Support Vector Machine"]
                selected_model = st.selectbox("🎯 Select a Classification Model", classification_models)

                model = None
                if selected_model == "Random Forest Classifier":
                    model = RandomForestClassifier()
                elif selected_model == "Logistic Regression":
                    model = LogisticRegression()
                elif selected_model == "Support Vector Machine":
                    model = SVC()

                if model is not None:
                    # Get X and Y variable names from the user using select boxes
                    x_variables = st.multiselect("📈 Select the X variables", data.columns)
                    y_variable = st.selectbox("🔍 Select the Y variable", data.columns)

                    if y_variable in x_variables:
                        st.error("❌ Invalid selection. Y variable must be different from X variables.")
                    else:
                        # Validate variable names and perform data splitting
                        if len(x_variables) > 0:
                            X = data[x_variables]
                            y = data[y_variable]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Calculate evaluation metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted")

                            st.write(f"📝 **Selected Classification Model**: {selected_model}")
                            st.write(f"📊 **X Variables**: {', '.join(x_variables)}")
                            st.write(f"📍 **Y Variable**: {y_variable}")
                            st.write(f"📈 **Accuracy**: {accuracy:.2f}")
                            st.write(f"📊 **F1 Score**: {f1:.2f}")
                        else:
                            st.error("❌ Invalid selection. Please ensure at least one X variable is selected.")
                else:
                    st.error("❌ An error occurred while selecting the model. Please try again.")

            elif problem_type == "Regression":
                st.subheader("📉 Regression Model Evaluation")

                regression_models = ["Random Forest Regressor", "Linear Regression", "Support Vector Machine"]
                selected_model = st.selectbox("🎯 Select a Regression Model", regression_models)

                model = None
                if selected_model == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif selected_model == "Linear Regression":
                    model = LinearRegression()
                elif selected_model == "Support Vector Machine":
                    model = SVR()

                if model is not None:
                    # Get X and Y variable names from the user using select boxes
                    x_variables = st.multiselect("📈 Select the X variables", data.columns)
                    y_variable = st.selectbox("🔍 Select the Y variable", data.columns)

                    if y_variable in x_variables:
                        st.error("❌ Invalid selection. Y variable must be different from X variables.")
                    else:
                        # Validate variable names and perform data splitting
                        if len(x_variables) > 0:
                            X = data[x_variables]
                            y = data[y_variable]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Calculate evaluation metrics
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            st.write(f"📝 **Selected Regression Model**: {selected_model}")
                            st.write(f"📊 **X Variables**: {', '.join(x_variables)}")
                            st.write(f"📍 **Y Variable**: {y_variable}")
                            st.write(f"📈 **Mean Absolute Error (MAE)**: {mae:.2f}")
                            st.write(f"📊 **R-squared (R²)**: {r2:.2f}")
                            st.write(f"📉 **Mean Squared Error (MSE)**: {mse:.2f}")
                        else:
                            st.error("❌ Invalid selection. Please ensure at least one X variable is selected.")
                else:
                    st.error("❌ An error occurred while selecting the model. Please try again.")
    else:
        st.warning("📂 Please upload a dataset in the 'Data Cleaning' step to continue.")

elif page == "Quick Links 🔗":
    st.header("🔗 Quick Links")
    st.write("Click on any link below to navigate to the respective application:")

    links = {
        "✍️ TextTrac": "https://texttrac-mmmj5kiucvh9muj66gekp4.streamlit.app/",
        "🧙‍♂️ Vision Wizard": "https://vision-wizard-durnsdepglthkhzx2peekt.streamlit.app/",
        "📂 Data Weaver": "https://aybzbegfeczo4yfievejqk.streamlit.app/",
        "🛠️ SkillSync": "https://skillsync-b8xdmwmdezbzf66qpbuj5j.streamlit.app/",
        "💰 TradeLens": "https://glseycvc2rbucwhk3esh85.streamlit.app/"
    }
    
    for name, url in links.items():
        st.link_button(name, url, use_container_width=True)

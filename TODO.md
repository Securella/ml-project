# Project To-Do List for Diabetes Readmission Prediction

### Instructions for Collaborators:
- As you complete each task, tick it by replacing `[ ]` with `[x]`.
- If you encounter any issues or questions, document them either in this file or submit an issue on GitHub.
- Commit and push your changes regularly to keep this project up to date.
- Use clear commit messages like: `"Completed Step 123"` or `"Implemented Random Forest Model"`.

---

## **Step 1: Data Loading and Initial Inspection**

- [x] Load csv files into pandas DataFrames.
- [x] Inspect the first few rows and check data integrity.
- [x] Check for missing values in the dataset (show result in code).
- [x] Verify the target variable (`readmitted`) for class imbalances.

---

## **Step 2: Exploratory Data Analysis ** ORKHAN

- [x] Generate summary statistics for numerical and categorical columns.
- [x] Visualize the distribution of categorical features (e.g., `gender`, `admission_type`).
- [x] Visualize the distribution of numerical features (e.g., `time_in_hospital`, `num_lab_procedures`).
- [x] Generate a correlation matrix and visualize using a heatmap.
- [x] Document key insights and patterns discovered during this step.

---

## **Step 3: Data Cleaning and Preprocessing** ORKHAN

### **3.1: Data Cleaning**

- [x] Drop irrelevant columns.
- [x] Handle missing values:
    - Impute missing categorical values with `'Unknown'` or the mode.
    - Impute missing numerical values with the median (or mean, depending on the situation).
- [x] (OPTIONAL) Remove extreme outliers in numerical columns (if that makes sense for our dataset).
- [x] Double chesk after each step/modification how the data looks like by visualizing, printing or showing first results in the table format.

### **3.2: Preprocessing**

- [ ] (DEPRECATED)Encode binary categorical variables (e.g., `gender`) with `LabelEncoder` (HAHAHA).
- [x] Apply one-hot encoding to multi-class categorical variables (e.g., `admission_type`).
- [x] Scale numerical features (e.g., `age`, `time_in_hospital`) using `StandardScaler` or similar.
- [x] Suggest if anything else should be done to make data quality amazing to work with.
- [x] Double chesk after each step/modification how the data looks like by visualizing, printing or showing first results in the table format.

---

## **Step 4: Train-Test Split** ORKHAN

- [x] Define the feature matrix `X` and the target variable `Y`.
- [x] Split the dataset into training and testing sets (80/20 split).

---

## **Step 5: Model Training**

- [ ] Train a Logistic Regression model.
- [ ] Train a Random Forest Classifier.
- [ ] Train an XGBoost Classifier.
- [ ] (OPTIONAL) Train a KNN.
- [ ] (OPTIONAL) Train a Naive Bayes.
- [ ] (OPTIONAL) Train an SVM.
- [ ] Document the initial performance of all models.

---

## **Step 6: Model Evaluation and Comparison**

- [ ] Evaluate each model using classification metrics (accuracy, precision, recall, F1 score, AUC).
- [ ] Compare models' performance.
- [ ] Generate and visualize ROC curves for each model.
- [ ] Plot feature importance for Random Forest and XGBoost models.

---

## **Step 7: Model Tuning (OPTIONAL)**

- [ ] Perform hyperparameter tuning on the Random Forest model using `GridSearchCV`.
- [ ] Perform hyperparameter tuning on the XGBoost model using `GridSearchCV`.
- [ ] Document any improvements in model performance after tuning.

---

## **Step 8: Web Integration (OPTIONAL)**

## **Step 8: Web Integration (Using Streamlit)**

### **8.1: Streamlit Web Application**
- [ ] Install Streamlit via pip (`pip install streamlit`).
- [ ] Create a Python file (e.g., `app.py`) for our app.
- [ ] Set up the Streamlit app to:
    - [ ] Display a form where users can input patient data (e.g., `age`, `time_in_hospital`).
    - [ ] Load the trained machine learning model.
    - [ ] Take inputs from the form, preprocess them, and pass them to the model for prediction.
    - [ ] Display the prediction result on the web page.
- [ ] Test the app locally using the command `streamlit run app.py`.
- [ ] Troubleshoot.
- [ ] Make it look good.  

---

## **Step 9: Deployment (OPTIONAL)**

### **9.1: Free Hosting Options**

#### **Option 1: Streamlit Cloud (Easiest)**
- [ ] Push your Streamlit app to a GitHub repository.
- [ ] Go to [Streamlit Cloud](https://share.streamlit.io/).
- [ ] Connect your GitHub account and select the repository containing your Streamlit app.
- [ ] Deploy the app to Streamlit Cloud (formerly Streamlit Sharing) and get a public URL.


#### **Option 2: AWS Free Tier with EC2**
- [ ] Launch an EC2 instance (Ubuntu).
- [ ] SSH into the instance and install necessary software (Python, Streamlit, etc.).
- [ ] Upload app files to the EC2 instance.
- [ ] Install dependencies on the server using `pip install -r requirements.txt`.
- [ ] Open HTTP/HTTPS ports in the security group for this EC2 instance.
- [ ] Run the Streamlit app on the EC2 instance and access it via the public IP address.
- [ ] (OPTIONAL) Set up NGINX as a reverse proxy to serve the app on port 80 (better practice).

---


### **Alternative: Flask Web Application**

- [ ] Set up a basic Flask app with necessary routes.
- [ ] Create a route for the home page displaying a form for patient data input.
- [ ] Create a route for handling form submissions and making predictions using the trained model.
- [ ] Build a simple HTML form to collect patient data (e.g., `age`, `time_in_hospital`).
- [ ] Integrate the trained machine learning model with Flask to make predictions based on form inputs.

---

## ** Deployment (OPTIONAL)**

- [ ] Prepare the Flask app for deployment on Heroku or any other cloud platform.
- [ ] Create a `Procfile` for Heroku deployment specifying the app entry point.
- [ ] Deploy the app to Heroku and verify that it is working.

---

## **Step 10: Report**

### Project Overview
- **Objective**: The goal of this project is to analyze and predict early readmission of diabetes patients within 30 days of discharge from hospitals using historical clinical data.
- **Dataset**: The dataset includes records from 130 US hospitals over a span of 10 years, comprising various features related to patient care.

### Methodology
- **Data Exploration**: Initial exploration of the dataset to understand the distribution of features and identify any anomalies or missing values.
- **Data Preprocessing**: Steps taken to clean the data, including handling missing values, normalization, and encoding categorical variables.
- **Feature Engineering**: Creating new features that may improve the model’s predictive power based on domain knowledge.
- **Model Selection**: Various classification models were evaluated, including Logistic Regression, Random Forest, and XGBoost, and so on.
- **Evaluation Metrics**: The models were assessed using metrics such as accuracy, precision, recall, and F1 score to ensure robustness.

### Challenges Faced
- **Data Quality Issues**: Encountered missing values and inconsistencies in the dataset which required significant preprocessing.
- **Model Overfitting**: Initial models showed signs of overfitting, necessitating the use of regularization techniques and cross-validation.
- **Deployment Issues**: Challenges in integrating the predictive model into a web application environment.

### Lessons Learned
- **Importance of Data Cleaning**: Comprehensive data cleaning and preprocessing are crucial for building effective predictive models.
- **Model Complexity**: Simpler models can sometimes outperform more complex ones if the data is not sufficient.
- **Other**: anything else of importance

---

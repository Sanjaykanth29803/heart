# ❤️ Heart Disease Prediction & Data Visualization App

##  Project Overview

The **Heart Disease Prediction App** is a machine learning based web application developed using **Python and Streamlit**.
This application predicts whether a patient is likely to have heart disease based on medical parameters and also provides **interactive data visualizations** to understand patterns in the dataset.

The system uses a **Random Forest Classifier** trained on the **Heart Disease Dataset** to make predictions and provides graphical insights into important health indicators.

---

##  Objectives

* Predict the **likelihood of heart disease** based on patient medical information.
* Provide **interactive data visualizations** for better understanding of heart disease factors.
* Demonstrate the application of **Machine Learning in healthcare analytics**.
* Build a **user-friendly dashboard** using Streamlit.

---

##  Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Plotly**

---

##  Machine Learning Model

The model used in this project is:

**Random Forest Classifier**

Why Random Forest?

* High accuracy
* Handles nonlinear data well
* Reduces overfitting
* Works well with structured medical datasets

### Data Preprocessing

Before training the model:

* Features are separated into **X (inputs)** and **y (target variable)**.
* Data is scaled using **StandardScaler** to normalize the values.
* The model is then trained using the scaled dataset.

---

##  Dataset Information

The dataset used is **heart.csv**, which contains medical attributes used for predicting heart disease.

### Features in Dataset

| Feature  | Description                              |
| -------- | ---------------------------------------- |
| age      | Age of the patient                       |
| sex      | Gender (1 = Male, 0 = Female)            |
| cp       | Chest pain type                          |
| trestbps | Resting blood pressure                   |
| chol     | Serum cholesterol level                  |
| fbs      | Fasting blood sugar (>120 mg/dl)         |
| restecg  | Resting electrocardiographic results     |
| thalach  | Maximum heart rate achieved              |
| exang    | Exercise induced angina                  |
| oldpeak  | ST depression induced by exercise        |
| slope    | Slope of peak exercise ST segment        |
| ca       | Number of major vessels                  |
| thal     | Thalassemia type                         |
| target   | Heart disease presence (1 = Yes, 0 = No) |

---

##  Application Features

###  Heart Disease Prediction

Users can input patient medical details such as:

* Age
* Gender
* Blood pressure
* Cholesterol level
* Heart rate
* Chest pain type
* Exercise induced angina

The model predicts whether:

*  The patient is likely to have heart disease
*  The patient is not likely to have heart disease

---

###  Data Visualizations

The application also provides **interactive visual analytics**, including:

**Heart Disease Count**

* Distribution of patients with and without heart disease.

**Age vs Heart Disease**

* Shows age distribution among patients.

**Chest Pain Type Distribution**

* Comparison of chest pain categories with disease occurrence.

**Correlation Heatmap**

* Displays relationships between all medical features.

**Max Heart Rate vs Age**

* Scatter plot showing relationship between heart rate and age.

These visualizations help users understand **patterns and correlations in the dataset**.

---

## Application Interface

The application contains **two main tabs**:

### Prediction

* User enters patient details.
* Machine learning model predicts heart disease risk.

### Data Visualizations

* Interactive charts and graphs for dataset exploration.

---

## Installation & Setup

### Clone the Repository

```bash
https://github.com/Sanjaykanth29803/heart.git
```

### Install Required Libraries

```bash
pip install streamlit pandas numpy seaborn matplotlib plotly scikit-learn
```

###  Run the Application

```bash
streamlit run app.py
```

---

Sample Output

The application provides:

* Patient prediction results
* Interactive charts
* Dataset insights
* Correlation heatmaps

---
Future Improvements

* Add **Deep Learning models**
* Deploy using **Streamlit Cloud / Vercel / AWS**
* Add **model accuracy evaluation metrics**
* Integrate **real-time patient data input**
* Improve **UI/UX design**

---

Author

**Sanjaykanth Chandran**
Final Year IT Student
Government College of Engineering, Erode (IRTT)

Data Analytics Enthusiast
Interested in Data Science, Machine Learning, and AI-driven solutions.

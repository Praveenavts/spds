import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Student Performance Data Science App",
    layout="wide"
)

st.title("Student Performance Data Analytics & Data Science Project")
st.write(
    "This project performs data analytics and predicts final marks using a simple Machine Learning model."
)

@st.cache_data
def load_data():
    csv_path = BASE_DIR / "student_performance.csv"
    return pd.read_csv(csv_path)

df = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset Preview",
        "Data Analytics Dashboard",
        "Machine Learning Prediction",
        "Model Performance"
    ]
)

if page == "Dataset Preview":
    st.header("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Average Final Marks", round(df["Final_Marks"].mean(), 2))

    pass_percentage = round((df["Result"].eq("Pass").mean()) * 100, 2)
    col3.metric("Pass Percentage", f"{pass_percentage}%")

    st.subheader("Statistical Summary")
    st.write(df.describe())

elif page == "Data Analytics Dashboard":
    st.header("Data Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Highest Marks", df["Final_Marks"].max())
    col2.metric("Lowest Marks", df["Final_Marks"].min())
    col3.metric("Average Attendance", round(df["Attendance"].mean(), 2))
    col4.metric("Average Study Hours", round(df["Study_Hours"].mean(), 2))

    st.subheader("Pass vs Fail Count")
    result_count = df["Result"].value_counts()
    st.bar_chart(result_count)

    st.subheader("Study Hours vs Final Marks")

    fig, ax = plt.subplots()
    ax.scatter(df["Study_Hours"], df["Final_Marks"])
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Marks")
    ax.set_title("Study Hours vs Final Marks")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig2, ax2 = plt.subplots()
    cax = ax2.matshow(corr)
    fig2.colorbar(cax)

    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="left")
    ax2.set_yticklabels(corr.columns)

    st.pyplot(fig2)

elif page == "Machine Learning Prediction":
    st.header("Predict Student Final Marks")

    X = df[
        [
            "Study_Hours",
            "Attendance",
            "Assignment_Score",
            "Previous_Marks"
        ]
    ]

    y = df["Final_Marks"]

    model = LinearRegression()
    model.fit(X, y)

    st.write("Enter student details below:")

    study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 5.0)
    attendance = st.slider("Attendance Percentage", 0.0, 100.0, 75.0)
    assignment_score = st.slider("Assignment Score", 0.0, 100.0, 70.0)
    previous_marks = st.slider("Previous Marks", 0.0, 100.0, 65.0)

    input_data = pd.DataFrame(
        [[study_hours, attendance, assignment_score, previous_marks]],
        columns=[
            "Study_Hours",
            "Attendance",
            "Assignment_Score",
            "Previous_Marks"
        ]
    )

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    st.subheader("Predicted Final Marks")
    st.success(f"{prediction:.2f}")

    if prediction >= 50:
        st.info("Prediction Result: Pass")
    else:
        st.warning("Prediction Result: Needs Improvement")

elif page == "Model Performance":
    st.header("Machine Learning Model Performance")

    X = df[
        [
            "Study_Hours",
            "Attendance",
            "Assignment_Score",
            "Previous_Marks"
        ]
    ]

    y = df["Final_Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error", round(mae, 2))
    col2.metric("R2 Score", round(r2, 2))

    st.subheader("Actual Marks vs Predicted Marks")

    result_df = pd.DataFrame(
        {
            "Actual Marks": y_test.values,
            "Predicted Marks": y_pred.round(2)
        }
    )

    st.dataframe(result_df)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")
    ax.set_title("Actual vs Predicted Marks")

    st.pyplot(fig)

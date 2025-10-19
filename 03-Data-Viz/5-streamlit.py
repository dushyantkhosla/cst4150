import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

# Set page title
st.set_page_config(page_title="Data Explorer Dashboard", layout="wide")

# Header
st.title("Data Explorer Dashboard")

# Sidebar
st.sidebar.header("Settings")

# Dropdown for dataset selection
dataset_option = st.sidebar.selectbox(
    "Choose a dataset",
    ("Iris", "Titanic")
)

# Main body
st.header(f"Exploring {dataset_option} Dataset")

# Function to load data based on selection
@st.cache_data
def load_data(dataset):
    if dataset == "Titanic":
        import seaborn as sns
        titanic = sns.load_dataset('titanic')
        return titanic
    elif dataset == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        return pd.DataFrame(iris.data, columns=iris.feature_names)

# Load the selected dataset
data = load_data(dataset_option)

# Display basic information about the dataset
st.subheader("Dataset Overview")
st.write(f"Shape of the dataset: {data.shape}")
st.write(f"Columns: {', '.join(data.columns)}")

# Display the first few rows of the dataset
st.subheader("Sample Data")
st.write(data.head())

# Basic statistics
st.subheader("Basic Statistics")
st.write(data.describe())

# Data visualization
st.subheader("Data Visualization")
st.write("Here you can add various plots and visualizations based on the selected dataset.")

# import titanic dataset from seaborn

# barchart of sepal length
if dataset_option == "Iris":
    data = load_data("Iris")
    st.subheader("Bar Chart: Sepal Length")
    

# Seaborn barchart
import matplotlib.pyplot as plt
st.subheader("Seaborn Bar Chart")
fig, ax = plt.subplots(figsize=(3, 4))

if dataset_option == "Iris":
    sns.barplot(x="species", y="sepal length (cm)", data=load_data("Iris"), ax=ax)
    plt.title("Average Sepal Length by Species")
elif dataset_option == "Titanic":
    sns.barplot(x="class", y="fare", data=load_data("Titanic"), ax=ax)
    plt.title("Average Fare by Passenger Class")
elif dataset_option == "Boston Housing":
    sns.barplot(x="RAD", y="PRICE", data="Boston Housing", ax=ax)
    plt.title("Average House Price by RAD (Index of accessibility to radial highways)")

plt.xlabel(ax.get_xlabel(), fontsize=12)
plt.ylabel(ax.get_ylabel(), fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)
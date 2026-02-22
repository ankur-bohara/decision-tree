import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Decision Tree")
st.title("ID3 Decision Tree Classifier")

# -----------------------------
# Helper Functions
# -----------------------------

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    probabilities = counts / len(col)
    return -np.sum([p * math.log2(p) for p in probabilities if p > 0])


def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    values = df[attr].unique()

    weighted_entropy = 0
    for v in values:
        subset = df[df[attr] == v]
        if len(subset) == 0:
            continue
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy


def id3(df, target, attrs):
    # If all target values same → return class
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # If no attributes left → return majority class
    if not attrs:
        return df[target].mode()[0]

    # Choose best attribute
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}

    for val in df[best].unique():
        subset = df[df[best] == val]

        if subset.empty:
            tree[best][val] = df[target].mode()[0]
        else:
            remaining_attrs = [a for a in attrs if a != best]
            tree[best][val] = id3(subset, target, remaining_attrs)

    return tree


def predict(tree, input_data):
    if not isinstance(tree, dict):
        return tree

    root = next(iter(tree))
    value = input_data.get(root)

    if value in tree[root]:
        return predict(tree[root][value], input_data)
    else:
        return "Unknown"


# -----------------------------
# Default Dataset
# -----------------------------

data_dict = {
    "outlook": ["sunny", "sunny", "overcast", "rain", "rain",
                "overcast", "sunny", "sunny", "overcast",
                "rain", "overcast", "overcast", "rain", "sunny"],
    "humidity": ["high", "normal", "high", "normal", "high",
                 "high", "normal", "normal", "normal",
                 "normal", "normal", "high", "high", "normal"],
    "playtennis": ["no", "yes", "yes", "yes", "no",
                   "yes", "yes", "yes", "no",
                   "yes", "no", "yes", "yes", "yes"]
}

df = pd.DataFrame(data_dict)

# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader("Upload CSV (Categorical Data Preferred)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

# Convert numeric columns to categorical (basic handling)
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.cut(df[col], bins=3, labels=["Low", "Medium", "High"])

st.subheader("Dataset")
st.dataframe(df)

# -----------------------------
# Target Selection
# -----------------------------

target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
features = [c for c in df.columns if c != target_col]

# -----------------------------
# Train Button
# -----------------------------

if st.button("Train Model"):
    if len(features) == 0:
        st.error("No feature columns available.")
    else:
        tree = id3(df, target_col, features)
        st.session_state["tree"] = tree
        st.success("Model Trained Successfully!")
        st.json(tree)

# -----------------------------
# Prediction Section
# -----------------------------

if "tree" in st.session_state:
    st.subheader("Make Prediction")

    user_input = {}
    for col in features:
        user_input[col] = st.selectbox(f"Select {col}", df[col].unique())

    if st.button("Predict"):
        result = predict(st.session_state["tree"], user_input)
        st.success(f"Prediction: {result}")

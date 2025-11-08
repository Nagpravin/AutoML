# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import pickle
from backend.Supervised import Classifier  # Make sure this matches your actual import
from backend.Preprocess import preprocess  # Adjust according to your actual preprocessing function

# Set Streamlit page config
st.set_page_config(page_title="ML Classification Comparison App", layout="wide")

# Function to build models and evaluate performance
def build_model(df):

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    st.markdown("**1.2. Dataset dimension**")
    st.write("X")
    st.info(X.shape)
    st.write("Y")
    st.info(Y.shape)

    st.markdown("**1.3. Variable details**:")
    st.write("X variable (first 20 are shown)")
    st.info(list(X.columns[:20]))
    st.write("Y variable")
    st.info(Y.name)

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-(split_size/100), random_state=seed_number)

    # Initialize the classifier
    clf = Classifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
    scores, predictions = clf.fit(X_train, X_test, Y_train, Y_test)

    st.subheader("2. Table of Model Performance")
    st.write(scores)

    st.subheader("4. Multiclass ROC Curve")
    plot_roc_curve_multiclass(clf, X_train, X_test, Y_train, Y_test)

    st.subheader("5. Model Selection and Download")
    model_selection_and_download(clf, X_train, Y_train)

# ROC Curve plotting for multiclass
def plot_roc_curve_multiclass(classifier, X_train, X_test, Y_train, Y_test):
    classes = np.unique(Y_train)
    Y_test_binarized = label_binarize(Y_test, classes=classes)
    n_classes = Y_test_binarized.shape[1]
    models = classifier.provide_models(X_train, X_test, Y_train, Y_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            Y_score = model.predict_proba(X_test)
        else:
            Y_score = model.decision_function(X_test)
            if Y_score.ndim == 1:
                Y_score = np.vstack((1 - Y_score, Y_score)).T

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test_binarized[:, i], Y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Model selection and downloading
def model_selection_and_download(classifier, X_train, Y_train):
    models = classifier.provide_models(X_train, X_train, Y_train, Y_train)  # Adjust according to your method signature
    model_names = list(models.keys())
    
    selected_model_name = st.selectbox("Select a model to download", options=model_names)
    
    if st.button("Download Model"):
        selected_model = models[selected_model_name]
        pickle_bytes = pickle.dumps(selected_model)  # WARNING: Be careful with pickle for untrusted sources
        st.download_button(label="Download Model", data=pickle_bytes, file_name=f"{selected_model_name}.pkl", mime="application/octet-stream")

# Streamlit UI setup
st.write("# Machine Learning Classification Web App")

with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

with st.sidebar.header("2. Set Parameters"):
    split_size = st.sidebar.slider("Data split ratio (% for Training Set)", 10, 90, 80, 5)
    seed_number = st.sidebar.slider("Set the random seed number", 1, 100, 42, 1)

def select_and_store_columns(data):
    st.write("### Select Columns")
    all_columns = data.columns.tolist()
    all_columns = ["Select All"] + all_columns  # Prepend "Select All" option
    selected_columns = st.multiselect('Select Columns', all_columns, default=all_columns[1:])
    if "Select All" in selected_columns:
        selected_columns = all_columns[1:]  # Exclude "Select All"
    return selected_columns

st.subheader("1. Dataset")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("**1.1. Glimpse of dataset**")
    st.write(df)

    # Select and store columns
    selected_columns = select_and_store_columns(df)
    if selected_columns:
        df = df[selected_columns]  # Filter DataFrame based on selected columns

    df = preprocess(df)
    build_model(df)
else:
    st.info("Awaiting for CSV file to be uploaded.")

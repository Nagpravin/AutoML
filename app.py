# line 315
# line 32
import streamlit as st
import pandas as pd
from backend.Supervised import Regressor
from backend.Preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import pickle
import sklearn
import xgboost


X_train = []
Y_train = []
st.set_page_config(
    page_title="The Machine Learning Algorithm Comparison App", layout="wide"
)


# @st.cache(suppress_st_warning=True)
# @st.cache_data
# @st.cache_resource
# @st.cache_data(allow_output_mutation=True)
# @st.cache_data(suppress_st_warning=True)
# @st.cache_data(experimental_allow_widgets=True)
def build_model(df):
    # df = df.loc[df.index <= 100]
    # df = df.loc[:100]
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

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1-(split_size/100), random_state=seed_number
    )
    reg = Regressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader("2. Table of Model Performance")

    st.write("Training set")
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, "training.csv"), unsafe_allow_html=True)

    st.write("Test set")
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, "test.csv"), unsafe_allow_html=True)

    st.subheader("3. Plot of Model Performance (Test set)")

    with st.markdown("**R-squared**"):
        predictions_test["R-Squared"] = [
            0 if i < 0 else i for i in predictions_test["R-Squared"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(
            y=predictions_test.index, x="R-Squared", data=predictions_test
        )
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt, "plot-r2-tall.pdf"), unsafe_allow_html=True)

    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, "plot-r2-wide.pdf"), unsafe_allow_html=True)

    with st.markdown("**RMSE (capped at 50)**"):
        predictions_test["RMSE"] = [
            50 if i > 50 else i for i in predictions_test["RMSE"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    st.markdown(imagedownload(plt, "plot-rmse-tall.pdf"), unsafe_allow_html=True)

    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, "plot-rmse-wide.pdf"), unsafe_allow_html=True)

    with st.markdown("**Calculation time**"):
        predictions_test["Time Taken"] = [
            0 if i < 0 else i for i in predictions_test["Time Taken"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(
            y=predictions_test.index, x="Time Taken", data=predictions_test
        )
    st.markdown(
        imagedownload(plt, "plot-calculation-time-tall.pdf"), unsafe_allow_html=True
    )

    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(
        imagedownload(plt, "plot-calculation-time-wide.pdf"), unsafe_allow_html=True
    )
    # with st.form(key='my_form'):
    #     st.header('3. Model Selection')
    #     mod = reg.fit_and_get_models(X_train, Y_train)
    #     model1 = st.selectbox(
    #     "Choose suitable model acc to metrics or of your choice",
    #     ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
    #         'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
    #     index=None,)
    #     # if st.button("Download Trained Model"):
    #         # download_link = model_download_link(loaded_model, f"Trained_Model_{model1}.pkl")
    #         # st.markdown(download_link, unsafe_allow_html=True)
    #     submit_button = st.form_submit_button(label='Choose')
    #     if model1 !=None:
    #         final=mod[model1]
    #         st.download_button(
    #         "Download Model",
    #         data=pickle.dumps(final),
    #         file_name="model.pkl",)

    st.header("3. Model Selection")
    mod = reg.fit_and_get_models(X_train, Y_train)
    model1 = st.selectbox(
        "Choose suitable model acc to metrics or of your choice",
        [
            "LinearSVR",
            "SVR",
            "RANSACRegressor",
            "KNeighborsRegressor",
            "LinearRegression",
            "RandomForestRegressor",
            "BaggingRegressor",
            "AdaBoostRegressor",
            "GradientBoostingRegressor",
            "XGBRegressor",
            "DecisionTreeRegressor",
        ],
        index=None,
    )
    # if st.button("Download Trained Model"):
    # download_link = model_download_link(loaded_model, f"Trained_Model_{model1}.pkl")
    # st.markdown(download_link, unsafe_allow_html=True)
    # submit_button = st.form_submit_button(label='Choose')
    if model1 != None:
        final = mod[model1]
        st.write("hurray")
        st.download_button(
            "Download Model",
            data=pickle.dumps(final),
            file_name="model.pkl",
        )

        # st.write(model1)
        # if model1 !=None:
        #     model1=mod[model1]
        #     loaded_model = load_model(model1)
        #     download_link = model_download_link(loaded_model, f"Trained_Model_{model1}.pkl")
        #     st.markdown(download_link, unsafe_allow_html=True)
    # st.markdown(modelselection())

    # st.subheader('4. Model Selection')
    # model1=st.selectbox('Choose suitable model acc to metrics or of your choice',
    # ('LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
    #     'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'),
    # index=None,
    # placeholder="Select model...")
    # if st.button("Choose Model"):

    #     model1 = st.radio(
    #     "Choose suitable model acc to metrics or of your choice",
    #     ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
    #         'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
    #     index=None,)
    # model1 = st.radio("Choose suitable model acc to metrics or of your choice",
    # ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
    #     'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor']
    # index=None)

    # st.write(model1)


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format="pdf", bbox_inches="tight")
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


# def load_model(selected_model_name):
#     model_name_valid = "".join(c for c in selected_model_name if c.isalnum() or c in ('_', '-'))
#     model_path = f"{selected_model_name}_model.pkl"
#     if os.path.exists(model_path):
#         with open(model_path, 'rb') as file:
#             loaded_model = pickle.load(file)
#         # st.success(f"Model {selected_model_name} loaded successfully.")
#     else:
#         # st.warning(f"Model {selected_model_name} not found. Creating a new model.")
#         # Create a new model here or handle it according to your needs
#         # For example, you can create a default model or load a pre-trained model
#         loaded_model = selected_model_name  # Replace this with your own logic

#         # Save the new model
#         with open(model_path, 'wb') as file:
#             pickle.dump(loaded_model, file)
#     return loaded_model

# def load_model(selected_model_name):
#     model_path = f"{selected_model_name}_model.pkl"
#     with open(model_path, 'rb') as file:
#         loaded_model = pickle.load(file)
#     return loaded_model

# def model_download_link(model, filename):
#     model_bytes = pickle.dumps(model)
#     model_str = repr(model).replace('\n', '')
#     model_b64 = base64.b64encode(model_bytes).decode()
#     href = f'<a href="data:application/octet-stream;base64,{model_b64}" download={filename}>Download {filename} Model</a>'
#     return href
# def model_download_link(model, filename):
#     # Serialize the model to bytes using pickle
#     model_bytes = pickle.dumps(model)

# Replace newline characters in the model's string representation


# Encode the model bytes in base64
# model_b64 = base64.b64encode(model_bytes).decode()

# # Create a download link with the encoded model
# href = f'<a href="data:application/octet-stream;base64,{model_b64}" download={filename}>Download {filename} Model</a>'
# return href


# def modelselection():
#     st.subheader('4. Model Selection')
#     if st.button("Choose Model"):

#         model1 = st.radio(
#         "Choose suitable model acc to metrics or of your choice",
#         ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
#             'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
#         index=None,)

#         st.write(model1)

# ---------------------------------#


st.write(
    """
# NO CODE MACHINE LEARNING WEB APP

"""
)


with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
   
with st.sidebar.header("2. Set Parameters"):
    split_size = st.sidebar.slider(
        "Data split ratio (% for Training Set)", 10, 90, 80, 5
    )
    seed_number = st.sidebar.slider("Set the random seed number", 1, 100, 42, 1)

# with st.sidebar.header('3. Model Selection'):
#     if st.sidebar.button("Choose Model"):

#         model1 = st.sidebar.radio(
#         "Choose suitable model acc to metrics or of your choice",
#         ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
#             'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
#         index=None,)
#         st.sidebar.write(model1)


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
    if st.button("Press to use Example Dataset"):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name="response")
        df = pd.concat([X, Y], axis=1)

        st.markdown("The Diabetes dataset is used as the example.")
        st.write(df.head(5))

        build_model(df)

# with st.expander("Model Selection"):
#     xx = st.radio(
#     "Choose suitable model acc to metrics or of your choice",
#     ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
#         'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
#     index=None,)
# st.write(xx)


# with st.sidebar.header('3. Model Selection'):
#     if st.sidebar.button("Choose Model"):

#         model1 = st.sidebar.radio(
#         "Choose suitable model acc to metrics or of your choice",
#         ['LinearSVR', 'SVR', 'RANSACRegressor','KNeighborsRegressor','LinearRegression','RandomForestRegressor',
#             'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor','DecisionTreeRegressor'],
#         index=None,)
#         st.sidebar.write(model1)


# if model1=='AdaBo ostRegressor':
#   output=sklearn.ensemble._weight_boosting.AdaBoostRegressor.fit(X_train,Y_train)
# elif model1=='BaggingRegressor':
#   output=sklearn.ensemble._bagging.BaggingRegressor.fit(X_train, Y_train)
# elif model1=='DecisionTreeRegressor':
#   output=sklearn.tree._classes.DecisionTreeRegressor.fit(X_train, Y_train)
# elif model1=='GradientBoostingRegressor':
#   output=sklearn.ensemble._gb.GradientBoostingRegressor.fit(X_train, Y_train)
# elif model1=='KNeighborsRegressor':
#   output=sklearn.neighbors._regression.KNeighborsRegressor.fit(X_train, Y_train)
# elif model1=='LinearRegression':
#   output=sklearn.linear_model._base.LinearRegression.fit(X_train, Y_train)
# elif model1=='LinearSVR':
#   output=sklearn.svm._classes.LinearSVR.fit(X_train, Y_train)
# elif model1=='RANSACRegressor':
#   output=sklearn.linear_model._ransac.RANSACRegressor.fit(X_train, Y_train)
# elif model1=='RandomForestRegressor':
#   output=sklearn.ensemble._forest.RandomForestRegressor.fit(X_train, Y_train)
# elif model1=='SVR':
#   output=sklearn.svm._classes.SVR.fit(X_train, Y_train)
# elif model1=='XGBRegressor':
#   output=xgboost.sklearn.XGBRegressor.fit(X_train, Y_train)
# else:
#   model1='none'

# if model1 != 'none':
# #   pickle.dump(model1,open('model.pkl','wb'))
#     st.download_button(
#     "Download Model",
#     data=pickle.dumps(model1),
#     file_name="model.pkl",
# )

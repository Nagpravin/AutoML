"""
Remove duplicate columns.
Remove duplicate tuples.
Null Value Transformation.
Replace data with median if no. of tuples are less (than 200-250).
Transform Outliers.
Encode Classes.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# # Reading csv file
# df = pd.read_csv("datasets\\titanic.csv", encoding="latin-1")

# # Data Cleaning module.
# # Drop duplicate columns
# df = df.loc[:, ~df.columns.duplicated()]
# # Drop duplicate tuples.
# df.drop_duplicates(inplace=True)
# # Drop columns which have too many null or no values.
# df = df.loc[
#     :,
#     ~df.columns.isin(
#         [i for i in df.columns if df[i].isnull().sum() >= len(df) * (0.25)]
#     ),
# ]

# # This Block of code finds the correct features for the prediction model.
# # Features which have more than 50% unique values are discarded and those with floating point values are kept.
# dependent = df.iloc[:, -1].name
# features = [
#     i
#     for i in df.columns.tolist()
#     if i != dependent
#     and df[i].nunique() / df[i].shape[0] <= 0.5
#     or df[i].dtype == float
# ]
# df = df[features + [dependent]]


# # Drop the tuples which have rare scenario results ie tuples which have class_count :: total_count ratio less than 5 percent.
# counts = df[dependent].value_counts()
# ratios = counts / len(df)
# threshold = 0.05
# to_drop = ratios[ratios < threshold].index.tolist()
# df = df[~df[dependent].isin(to_drop)]

# # Encode the categorical columns
# le = LabelEncoder()
# for col in df.columns:
#     if df[col].dtype == object:
#         df[col] = le.fit_transform(df[col])

# # Now that all relevant categorized data are encoded, we clean the data.
# # Fill empty rows with mean. These rows can be dropped using dropna alternatively.
# # df.dropna(inplace=True)
# df.fillna(df.median(), inplace=True)
# # Remove Outlier values.
# df = df[(df - df.mean()).abs() <= 3 * df.std()]


# # Data Reduction Module.

# # Dimensionality Reduction.

# # Numerosity Reduction.
# # Reduce dataset so that it contains a maximum of 5000 tuples that are randomly shuffled.
# df = df.sample(n=5000 if len(df.index) > 5000 else len(df.index), random_state=42)


# df.to_csv("Output.csv", index=False)


def preprocess(df):
    df.dropna(how="all", inplace=True, axis=1)
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop duplicate tuples
    df.drop_duplicates(inplace=True)

    # Drop columns which have too many null or no values
    df = df.loc[
        :,
        ~df.columns.isin(
            [i for i in df.columns if df[i].isnull().sum() >= len(df) * (0.25)]
        ),
    ]

    # Discard features with more than 50% unique values and keep those with floating point values
    dependent = df.iloc[:, -1].name
    features = [
        i
        for i in df.columns.tolist()
        if (df[i].nunique() / df[i].shape[0] <= 0.5 or df[i].dtype == float)
        and i != dependent
    ]
    categorical = [i for i in features if df[i].dtype == object]
    df = df[features + [dependent]]

    # Drop tuples which have class_count :: total_count ratio less than 5 percent
    if df[dependent].dtype != float:
        counts = df[dependent].value_counts()
        ratios = counts / len(df)
        threshold = 0.05
        to_drop = ratios[ratios < threshold].index.tolist()
        df = df[~df[dependent].isin(to_drop)]

    # Encode the categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])

    # Remove outlier values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols][
        ((df[num_cols] - df[num_cols].mean()).abs() <= 3 * df[num_cols].std())
    ]

    # Fill empty rows with median / Remove them
    df.replace("", np.nan, inplace=True)
    df.dropna(how="any", inplace=True)

    # Reduce dataset so that it contains a maximum of 5000 tuples that are randomly shuffled
    df = df.sample(n=5000 if len(df.index) > 5000 else len(df.index), random_state=0)

    return df

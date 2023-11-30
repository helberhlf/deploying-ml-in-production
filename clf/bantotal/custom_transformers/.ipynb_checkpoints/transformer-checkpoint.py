#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

# Importing classes and libraries needed for the data pre-processing step.
from sklearn import preprocessing

# Importing pipelines
from sklearn.pipeline import  Pipeline
from sklearn.compose import (
    ColumnTransformer,
)
# Importing Classes for custom transformers
from sklearn.base import (
    BaseEstimator, TransformerMixin
)
# Importing libraries for dimensionality reduction
from sklearn.decomposition import PCA

# Importing libray for extracting Column Names from the ColumnTransformer
from skimpy import clean_columns
from sklearn.utils.validation import check_is_fitted
#-------------------------------------------------------

# Functions for transformations
""""
Credit :
https://practicaldatascience.co.uk/data-science/how-to-convert-pandas-dataframe-column-values
"""
# Convert Pandas column values to float
def cols_to_float(df, columns):
    """Convert selected column values to float and return DataFrame.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.
    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].astype(float)

    return df

# Convert Pandas column values to int
def cols_to_int(df, columns):
    """Convert selected column values to int and return DataFrame.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.
    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].astype(int)

    return df

# Convert Pandas column values to datetime
def cols_to_datetime(df, columns):
    """Convert selected column values to datetime and return DataFrame.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.
    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = pd.to_datetime(df[col], format='%Y%m%d')

    return df

# Convert Pandas column values to negatives
def cols_to_negative(df, columns):
    """Convert selected column values to negative and return DataFrame.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.
    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col] * -1

    return df

# Convert Pandas column values to log
def cols_to_log(df, columns):
    """Transform column data with log and return new columns of prefixed data.
    For us with data where the column values do not include zeroes.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log_' + col] = np.log(df[col])

    return df

# Convert Pandas column values to log+1
def cols_to_log1p(df, columns):
    """Transform column data with log+1 and return new columns of prefixed data.
    For use with data where the column values include zeroes.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log1p_' + col] = np.log(df[col] + 1)

    return df

# Convert Pandas column values to log max root

""""
The log max root is another useful transformation. The below function will take the dataframe and a list of columns
and will then convert each value to a log max root value. It is used when the data contains zeroes.
"""
def cols_to_log_max_root(df, columns):
    """Convert data points to log values using the maximum value as the log max and return new columns of prefixed data.
    For use with data where the column values include zeroes.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        log_max = np.log(df[col].max())
        df['logmr_' + col] = df[col] ** (1 / log_max)

    return df

# Convert Pandas column values to their hyperbolic tangent or tanh
def cols_to_tanh(df, columns):
    """Transform column data with hyperbolic tangent and return new columns of prefixed data.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['tanh_' + col] = np.tanh(df[col])

    return df

# Convert Pandas column values to 0 to 1 using sigmoid

"""
There are various ways to scale data points to values between 0 and 1. The sigmoid function is one way to achieve this. 
The below function will take the dataframe and a list of columns and will then convert each value to a sigmoid value
"""
def cols_to_sigmoid(df, columns):
    """Convert data points to values between 0 and 1 using a sigmoid function and return new columns of prefixed data.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        e = np.exp(1)
        y = 1 / (1 + e ** (-df[col]))
        df['sig_' + col] = y

# Convert Pandas column values to 0 to 1 using cube root
"""
The cube root will also convert data points to values between 0 and 1. 
The below function will take the dataframe and a list of columns and will then convert each value to a cube root value.
"""
def cols_to_cube_root(df, columns):
    """Convert data points to their cube root value so all values are between 0-1 and return new columns of prefixed data.
    Args:
        df: Pandas dataframe.
        columns: List of columns to transform.
    Returns:
        Original dataframe with additional prefixed columns.
    """

    for col in columns:
        df['cube_root_' + col] = df[col] ** (1 / 3)

    return df

# Convert Pandas column values to 0 to 1 using normalized cube root
"""
The next way of standardising values is the normalized cube root, which will also return values between 0 and 1. 
The below function will take the dataframe and a list of columns and will then convert each value to a normalized cube 
root value by determining the min() and max() values for each column.
"""
def cols_to_cube_root_normalize(df, columns):
    """Convert data points to their normalized cube root value so all values are between 0-1 and return new columns of prefixed data.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['cube_root_' + col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) ** (1 / 3)

    return df

# Convert Pandas column values to 0 to 1 using normalization
"""
Normalization is perhaps the most common way to scale data to values between 0 and 1. 
The below function will take the dataframe and a list of columns and will then convert each value to a normalized value by determining the min() and max() values for each column.
"""
def cols_to_normalize(df, columns):
    """Convert data points to values between 0 and 1 and return new columns of prefixed data.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['norm_' + col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df

# Convert Pandas column values to log+1 normalized
"""
When zeroes are present in the columns, you can also use normalization with a log+1 transformation. Again, this will return values between 0 and 1. 
The below function will take the dataframe and a list of columns and will then convert each value to a log+1 normalized value by determining the min() and max() values for each column.
"""
def cols_to_log1p_normalize(df, columns):
    """Transform column data with log+1 normalized and return new columns of prefixed data.
    For use with data where the column values include zeroes.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log1p_norm_' + col] = np.log((df[col] - df[col].min()) / (df[col].max() - df[col].min()) + 1)

    return df

# Convert Pandas column values to their percentile linearized value

"""
Percentile linearization will rank each data point by its percentile. To do this we can use the Pandas rank() function and a lambda function. 
The below function will take the dataframe and a list of columns and will then convert each value to a percentile linearized value.
"""
def cols_to_percentile(df, columns):
    """Convert data points to their percentile linearized value and return new columns of prefixed data.
    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.
    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['pc_lin_' + col] = df[col].rank(method='min').apply(lambda x: (x - 1) / len(df[col]) - 1)

    return df
#-------------------------------------------------------

# Get the column names from thea ColumnTransformer containing transformers & pipelines
""""
Credit :
https://github.com/scikit-learn/scikit-learn/issues/12525
https://johaupt.github.io/blog/columnTransformer_feature_names.html
https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/compose/_column_transformer.py#L345
"""
def get_column_names_from_ColumnTransformer(column_transformer, clean_column_names=True, verbose=True):
    """
Reference: Kyle Gilde: https://github.com/kylegilde/Kaggle-Notebooks/blob/master/Extracting-and-Plotting-Scikit-Feature-Names-and-Importances/feature_importance.py
Description: Get the column names from the a ColumnTransformer containing transformers & pipelines
Parameters
----------
verbose: Bool indicating whether to print summaries. Default set to True.
Returns
-------
a list of the correct feature names
Note:
If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns,
it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns
to the dataset that didn't exist before, so there should come last in the Pipeline.
Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525
"""

    assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
    check_is_fitted(column_transformer)
    new_feature_names, transformer_list = [], []

    for i, transformer_item in enumerate(column_transformer.transformers_):
        transformer_name, transformer, orig_feature_names = transformer_item
        orig_feature_names = list(orig_feature_names)

        if len(orig_feature_names) == 0:
            continue
        if verbose:
            print(f"\n\n{i}.Transformer/Pipeline: {transformer_name} {transformer.__class__.__name__}\n")
            print(f"\tn_orig_feature_names:{len(orig_feature_names)}")
        if transformer == 'drop':
            continue
        if isinstance(transformer, Pipeline):
            # if pipeline, get the last transformer in the Pipeline
            transformer = transformer.steps[-1][1]
        if hasattr(transformer, 'get_feature_names_out'):
            if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
                names = list(transformer.get_feature_names_out(orig_feature_names))
            else:
                names = list(transformer.get_feature_names_out())
        elif hasattr(transformer, 'get_feature_names'):
            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:
                names = list(transformer.get_feature_names(orig_feature_names))
            else:
                names = list(transformer.get_feature_names())

        elif hasattr(transformer, 'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer, 'features_'):
            # is this a MissingIndicator class?
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]
        else:
            names = orig_feature_names
        if verbose:
            print(f"\tn_new_features:{len(names)}")
            print(f"\tnew_features: {names}\n")

        new_feature_names.extend(names)
        transformer_list.extend([transformer_name] * len(names))
    transformer_list, column_transformer_features = transformer_list, new_feature_names

    if clean_column_names:
        new_feature_names = list(clean_columns(pd.DataFrame(columns=new_feature_names)).columns)

    return new_feature_names

# Creating a transformer for Label Encoder Transformation
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """def __init__(self, columns):
        self.columns = columns"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Create an instancia of class LabelEncoder
        le = preprocessing.LabelEncoder()

        # Apply the encoder to categorical variables
        for i in range(0, X.shape[1]):
            if X.dtypes[i] == 'object':
                X[X.columns[i]] = le.fit_transform(X[X.columns[i]])
        return X
#-------------------------------------------------------

# Function para computer PCA
def computePCA(data):
    pca = PCA()  # Compute PCA
    data_pca    = pca.fit_transform(data)  # Fit and transform data
    eigenvalues = pca.explained_variance_  # Get eigenvalues
    eigenvalues = np.round(eigenvalues, 5)  # Round off eigenvalues
    components  = pca.n_components_  # Get the number components
    total_var   = pca.explained_variance_ratio_.sum() * 100  # Get the total explained variend

    return data_pca, eigenvalues, components, total_var

def computePCA_v2(feature_df,features):
    pca = PCA()
    pca.fit(feature_df[features])
    components = pca.fit_transform(feature_df[features])
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Return the PCA calculation, components and load variable
    return pca, components, loadings
#-------------------------------------------------------

# Creating a transformer to remove unwanted columns
class DropColumns(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # First we copy the input Dataframe 'X'
        data = X.copy()
        # We return a new dataframe without the unwanted columns
        return data.drop(labels=self.columns, axis=1)
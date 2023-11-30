#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

# Importing classes to calculate some statistics.
import scipy.special as scsp
from scipy import stats
from scipy.stats import(
    gamma,
    norm,
    shapiro,
    stats,
    t,
)
# Importing classes and libraries needed for the data pre-processing step.
from sklearn.preprocessing import (
    minmax_scale,normalize,
    LabelEncoder, OneHotEncoder,
)
from IPython.display import display
#-------------------------------------------------------

# Using Global Constants Defining Named Constants
DECIMAL_LIMIT = 6
#-------------------------------------------------------

# Create a chi-square test function
"""""
Credit:
https://towardsdatascience.com/statistics-in-python-using-chi-square-for-feature-selection-d44f467ca745
"""""
def chi2(df, col1, col2):
    # Create the contingency table
    df_cont = pd.crosstab(index=df[col1], columns=df[col2])
    #print("-" * 40, "Contingency table", "-" * 40)
    #display(df_cont)

    # Calculate degree of freedom
    degree_f = (df_cont.shape[0] - 1) * (df_cont.shape[1] - 1)

    # Sum up the totals for row and columns
    df_cont.loc[:, 'Total'] = df_cont.sum(axis=1)
    df_cont.loc['Total'] = df_cont.sum()
    print("-" * 40, "Observed (O)", "-" * 40)
    display(df_cont)

    # Create the expected value dataframe
    df_exp = df_cont.copy()
    df_exp.iloc[:, :] = np.multiply.outer(df_cont.sum(1).values, df_cont.sum().values) / df_cont.sum().sum()
    print('---Expected (E)---')
    display(df_exp)

    # Calculate chi-square values
    df_chi2 = ((df_cont - df_exp) ** 2) / df_exp
    df_chi2.loc[:, 'Total'] = df_chi2.sum(axis=1)
    df_chi2.loc['Total'] = df_chi2.sum()

    print("-" * 40, "Chi-Square", "-" * 40)
    display(df_chi2)
    # Get chi-square score
    chi_square_score = df_chi2.iloc[:-1, :-1].sum().sum()

    # Calculate the p-value
    p = stats.distributions.chi2.sf(chi_square_score, degree_f)

    return chi_square_score, degree_f, p
#-------------------------------------------------------

# Creating a function to calculate the correlation coefficient between variables with ajust of threshold
"""
Credit:
https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
"""
def corr_cols(df,thresh):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    dic = {'Feature_1':[],'Feature_2':[],'Coef.Pearson':[]}
    for col in upper.columns:
        corl = list(filter(lambda x: x >= thresh, upper[col] ))
        #print(corl)
        if len(corl) > 0:
            inds = [round(x,4) for x in corl]
            for ind in inds:
                #print(col)
                #print(ind)
                col2 = upper[col].index[list(upper[col].apply(lambda x: round(x,4))).index(ind)]
                #print(col2)
                dic['Feature_1'].append(col)
                dic['Feature_2'].append(col2)
                dic['Coef.Pearson'].append(ind)
    return pd.DataFrame(dic).sort_values(by="Coef.Pearson", ascending=False)

# Function for remove collinear features in a dataframe
def remove_collinear_features(df_model, target_var, threshold, verbose):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold and which have the least correlation with the target (dependent) variable. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        df_model: features dataframe
        target_var: target (dependent) variable
        threshold: features with correlations greater than this value are removed
        verbose: set to "True" for the log printing

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = df_model.drop(target_var, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold, coef of Pearson ranges from -1 to 1.
            if val >= threshold:
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))

                """
                 If we will add abs( ) function while calculating the correlation value between target and feature, we will not see negative correlation value. 
                 It is important because when we have negative correlation code drops smaller one which has stronger negative correlation value. 
                """
                # col_value_corr = abs(df_model[col.values[0]].corr(df_model[target_var]))
                # row_value_corr = abs(df_model[col.values[0]].corr(df_model[target_var]))

                col_value_corr = df_model[col.values[0]].corr(df_model[target_var])
                row_value_corr = df_model[row.values[0]].corr(df_model[target_var])
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df_model = df_model.drop(columns=drops)

    print("dropped columns: ")
    print(list(drops))
    print("-----------------------------------------------------------------------------")
    print("used columns: ")
    print(df_model.columns.tolist())

    return df_model
#-------------------------------------------------------

# Creating functions for normality testing

# Function to calculate the normality test using the Shapiro Test method
def shapiro_test(df):
    # Named constants to represent the significance level
    #ALPHA_99_9 = 0.001
    #ALPHA_99 = 0.01
    #ALPHA_95 = 0.05

    # The p-value approach
    print("\nApproach : The p-value approach to hypothesis testing in the decision rule\n")

    CONCLUSION = "Failed to reject the null hypothesis."  # The null hypothesis cannot be rejected
    # loop for to through the desired columns for testing
    for cols in df:
        print(f'Variable: {[cols]}\n')
        # Calculation result
        sw, p_value = shapiro(df[cols])
        # Nested conditional decision structure to test test significance level
        if p_value <= 0.001:
            CONCLUSION = 'Null Hypothesis is rejected.'
            print(f'H1 is accepted, which is in the 99.9% region of acceptance -> {CONCLUSION}')
        elif p_value <= 0.01:
            CONCLUSION = 'Null Hypothesis is rejected.'
            print(f'H1 is accepted, which is in the 99% region of acceptance -> {CONCLUSION}')
        elif p_value <= 0.05555:
            CONCLUSION = 'Failed to reject the null hypothesis'
            print(f'H0 is accepted, which is in the 95% region of acceptance -> {CONCLUSION}')
        else:
            print(f'It is assumed that the data is normally distributed -> {CONCLUSION}')

        print(f'SW-statistic: {sw}\t p-value: {p_value:,.5f}\n')
        print('-' * 100)
#-------------------------------------------------------

# Functions to Calculate z score

def calc_zscore(df, categorical, numeric):
    # Group by
    df = df.groupby([categorical])[numeric].sum().reset_index().sort_values(by=[numeric], ascending=False).reset_index(drop=True)

    # Selecting the variable
    x = df.loc[:, [numeric]]
    # calculate z score
    df['z_score'] = (x - x.mean()) / x.std()

    # Assigning colors
    df['colors'] = ['red' if x < 0 else 'green' for x in df['z_score']]

    # Destandardizing Z Score to Percentage
    def z2p(z):
        """From z-score return p-value."""
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))

    # Function composition
    des_z = round(z2p(df['z_score']) * 100, 2)
    # des_z2 = norm.cdf(df['z_score'])*100

    # Assign percentage value of each z score
    # df['relative frequency (%)'] = des_z
    df['relative_freq'] = des_z

    # Ordering
    df.sort_values([numeric], inplace=True)

    return df

"""
Credit:
https://stackoverflow.com/questions/3496656/convert-z-score-z-value-standard-score-to-p-value-for-normal-distribution-in
"""
def get_p_value_normal(z_score: float) -> float:

    """get p value for normal(Gaussian) distribution

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(norm.sf(z_score),DECIMAL_LIMIT)

def get_p_value_t(z_score: float) -> float:
    """get p value for t distribution

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(t.sf(z_score), DECIMAL_LIMIT)

def get_p_value_chi2(z_score: float) -> float:
    """get p value for chi2 distribution

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(chi2.ppf(z_score,), DECIMAL_LIMIT)
#-------------------------------------------------------

# Creating a function to capture relative and cumulative frequencies
def relative_and_cumulative(df, categorical, numeric):
    # Groyp by
    df = df.groupby([categorical])[numeric].sum().reset_index().sort_values(by=numeric, ascending=False)
    # Capturing the relative and cumulative frequency of sales by continent
    df["fr"] = df[numeric] / df[numeric].sum() * 100
    df["Fr"] = df[numeric].cumsum() / df[numeric].sum() * 100
    """
    df["Percent"] = round(df[numeric] / df[numeric].sum() * 100, 2)
    df["Cumulative Percent"] = round(df[numeric].cumsum() / df[numeric].sum() * 100, 2 )
    """
    # Return dataframe with their relative and cumulative frequency
    return df.reset_index(drop=True)

# Create function gamma probability distribution maximization
def dist_gamma(df, categorical, numeric):
    # Group by
    df = df.groupby([categorical])[numeric].sum().reset_index().sort_values([numeric], ascending=False)
    #Capturing the cumulative sum
    df["cumulative_sum"] = df[numeric].cumsum()

    # Capturing Freq relative and Cumulative
    df["fr"] = df[numeric] / df[numeric].sum() * 100
    df["Fr"] = df[numeric].cumsum() / df[numeric].sum() * 100

    shape, loc, scale = gamma.fit(df[numeric])
    x = np.linspace(df.cumulative_sum.min(), df.cumulative_sum.max())
    predictions_gamma = gamma.pdf(x=x, a=shape, loc=loc, scale=scale)

    # Create dataframe with forecasts
    df_pred = pd.DataFrame(predictions_gamma, columns=['Predictions'])
    df_pred['Actual'] = df.fr

    # Normalize the data
    df_min_max = minmax_scale(df_pred)
    df_pred = pd.DataFrame(df_min_max, columns=['Predictions', 'Actual']).fillna(0)
    # return probability distribution maximization
    return df_pred

# Create a confidence interval function with 95% significance
def ci_95(df, numeric1, numeric2):
    # We created a grouped version, with calculated mean and standard deviation with custom aggregation function
    agg_func_custom_count = {numeric2: ["mean", "std", "count"]}
    df = df.groupby([numeric1]).agg(agg_func_custom_count)
    df = df.droplevel(axis=1, level=0).reset_index()

    # Calculate the 95% confidence interval.
    df['ci'] = 1.96 * df['std'] / np.sqrt(df['count'])
    df['ci_lower'] = df['mean'] - df['ci']
    df['ci_upper'] = df['mean'] + df['ci']

    # If you have nan values replace with 0
    return df.fillna(0)

# Defining a function to calculate the confidence interval, accuracy and error of the models.
def ci(score, nElements, ci=.95):

    # Define o Z-score, a ser utilizado para o intervalo de confiança selecionado.
    z = {
        .90: 1.645,  # Para Intervalos com 90% de Confiança.
        .95: 1.96,   #  Para Intervalos com 95% de Confiança.
        .98: 2.326,  # Para Intervalos com 98% de Confiança.
        .99: 2.576,  # Para Intervalos com 99% de Confiança.
        .995: 2.807, # Para Intervalos com 99.5% de Confiança.
        .999: 3.291  # Para Intervalos com 99.9% de Confiança.
    }

    # Calculando o range de variação do Intervalo.
    interval = z.get(ci) * np.sqrt((score * (1 - score)) / nElements)

    # retornando do intervalo de confiança.
    return score - interval, score + interval

# calculate confidence interval for the RMSE
def ci_rmse(y_pred, y_test, confidence = .95):

    squared_errors = (y_pred - y_test) ** 2
    return np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),scale=stats.sem(squared_errors)))

# Outlier treatment use using
def outliers(x,qt_inf, qt_sup):
    """
    A função retorna um vetor booleano: True se o elemento for um outlier. Falso, caso contrário.
    Agora, para substituir os outliers superiores e inferiores, vamos escrever outra pequena função e aplicá-la em todo o dataframe:
    """
    return np.abs(x - x.median()) > 1.5 * (x.quantile(qt_sup) - x.quantile(qt_inf))
def replace(x, th_inf, th_sup,):
    """
    Substitua os valores outliers mais altos pelo percentil 95 e os valores outliers mais baixos pelo percentil 5
    """
    out = x[outliers(x)]  # é um vector
    return x.replace(to_replace=[out.min(), out.max()],
                     value=(np.percentile(x, th_inf), np.percentile(x, th_sup)))
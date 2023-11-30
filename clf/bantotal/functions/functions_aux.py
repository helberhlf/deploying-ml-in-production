#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing libraries needed for Operating System Manipulation in Python
import platform, psutil

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd
#-------------------------------------------------------

# Function for System Information
# Credits: https://thepythoncode.com/article/get-hardware-system-information-python
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
def info_system():
    print("=" * 40, "System Information", "=" * 40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")

    # let's print CPU information
    print("=" * 40, "CPU Info", "=" * 40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()

    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")

    # Memory Information
    print("=" * 40, "Memory Information", "=" * 40)

    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")

# Functions to calculate differences in datasets
def summary_quick_in_between_datasets(x, y):
    print(f'First dataset, get the number of rows and columns: {x.shape}')
    print(f'Get the number of elements: {x.size}', end=' ')
    print('\n')
    print(f'Second dataset, get the number of rows and columns: {y.shape}')
    print(f'Get the number of elements: {y.size}')

    print('-' * 70)

    rows, cols = abs(x.shape[0] - y.shape[0]), abs(abs(x.shape[1] - y.shape[1]))
    print(f'Difference between datasets in rows: {rows} and columns: {cols}')

# Function to count null and missing values
# Credit: https://github.com/chris1610/sidetable
def missing(df, clip_0=False, style=False, tot=False):
        """ Build table of missing data in each column.

            clip_0 (bool):     In cases where 0 counts are generated, remove them from the list
            style (bool):     Apply a pandas style to format percentages

        Returns:
            DataFrame with each Column including total Missing Values, Percent Missing
            and Total rows
        """
        missing = pd.concat(
            [df.isna().sum(),
             df.isna().mean().mul(100)],
            axis='columns').rename(columns={
            0: 'missing',
            1: 'percent'
        })
        total = df.isnull().sum().sum()
        total_perc = (df.isnull().sum() / df.shape[0]) * 100

        print(f'\nTotal missing:  {total}')
        print(f'\nTotal in percentage terms: {round(total_perc[total_perc > 0], 3).sum()}%')
        missing['total'] = len(df)
        if clip_0:
            missing = missing[missing['missing'] > 0]

        results = missing[['missing', 'total',
                           'percent']].sort_values(by=['missing'],
                                                   ascending=False)
        if style:
            format_dict = {
                'percent': '{:.2f}%',
                'total': '{0:,.0f}',
                'missing': '{0:,.0f}'
            }
            return results.style.format(format_dict)
        else:
            return results

def missing_values(df):
    """For each column with missing values and  the missing proportion."""
    data = [(col, df[col].isna().sum() / len(df) * 100)
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing', ]

    # Create dataframe with values missing in an ordered way
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')

    # Return dataframe the values missing
    return missing_df

def unique_nan(x):
    return x.nunique(dropna=False)
def count_nulls(x):
    return x.size - x.count()



import matplotlib.pyplot as plt

'''Functions used to summarize our dataset.'''



# ------------ print functions ---------------

def print_summary(data):
    '''Prints a summary of the dataset.
    
    args:
    - data: pandas DataFrame
    '''
    print(f'The dataset contains,' +
          f'\n\033[1m{len(data['variable'])}\033[0m records' +
          f'\n\033[1m{len(data.id.unique())}\033[0m participants' +
          f'\n\033[1m{len(data.variable.unique())}\033[0m variables' +
          f'\n\033[1m{data.value.isnull().sum()}\033[0m missing values')


def print_variable_list(data, sorted=False):
    '''Prints a list of variables in the dataset.
    
    args:
    - data: pandas DataFrame
    - sorted: bool, default=False
    '''
    print('The variables are,')
    variables = [x for x in data["variable"].unique()]
    variables.sort(key=lambda x: x[0])  if sorted else None
    for i, variable in enumerate(variables):
        print(f'[{i}] : {variable}')


def print_variable_ranges(data):
    '''Prints the range of values for each variable in the dataset.
    
    args:
    - data: pandas DataFrame
    '''
    print('The range of values for each variable is,')
    for variable in data["variable"].unique():
        values = data[data["variable"] == variable]["value"]
        print(f'{variable} : [{values.min():.2f}, {values.max():.2f}]')


def print_modes(data):
    '''Prints the mode of values for each variable in the dataset.
    
    args:
    - data: pandas DataFrame
    '''
    print('For each variable, the modes is/are,')
    for variable in data["variable"].unique():
        values = data[data["variable"] == variable]["value"]
        print(f'{variable} : {[round(val, 2) for val in values.mode().tolist()]}')


def print_variable_distributions(data):
    '''Prints the distribution of values for each variable in the dataset.
    
    args:
    - data: pandas DataFrame
    '''
    print('The distribution of each variable is,')
    max_var_len = max(len(str(var)) for var in data["variable"].unique()) + 1

    header = f'| {"Variable".ljust(max_var_len)} | {"Mean (Std)".center(16)} | {"Median".center(8)} |'
    print(header)
    print('-' * len(header))
    
    for variable in data["variable"].unique():
        values = data[data["variable"] == variable]["value"]
        mean = values.mean()
        std = values.std()
        median = values.median()
        mode = values.mode().iloc[0]

        line = f'| {variable.ljust(max_var_len)}' + \
               f'| {mean:7.2f} ({std:5.2f})'.center(20) + \
               f'| {median:7.2f}'.center(10) + ' |'

        print(line)



# ----------------- plot functions --------------------

def plot_barchart(x_data, y_data, x_label, y_label, figsize, color, rotate_x_labels=False, show=False, export=False, add_values_to_bars=None, add_values_to_bars_offset=None, y_lim=None, title=None):
    
    # steez
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=14) if title else None
    
    plt.xlabel(x_label, fontsize=12)
    plt.xticks(rotation=45, ha='right')  if rotate_x_labels else None
    
    plt.ylabel(y_label, fontsize=12)
    plt.ylim(y_lim)  if y_lim else None
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # barz
    bars = plt.bar(x_data, y_data, color=color, edgecolor='black') 
    
    # add cool numbers to barz
    if add_values_to_bars:
        for bar, count in zip(bars, y_data):
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                count + add_values_to_bars_offset, 
                str(count), 
                rotation=90, 
                ha='center', 
                # color='white',
                fontsize=12
            )


    # resz
    plt.show() if show else None
   
    # TODO: export shit if specified


# ----------- debugging ---------------

if __name__ == '__main__':
    
    import os
    import pandas as pd

    PATH = os.getcwd()
    DATA = os.path.join(PATH, 'data/mood_smartphone.csv')

    data = pd.read_csv(DATA)


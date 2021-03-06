# coding=utf-8
import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler
from matplotlib import pyplot # hist plot group
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')
path = ''
########################################## data observing ########################################
def missing_data(file, save):
    nulls = np.sum(file.isnull())
    nullcols = nulls.loc[(nulls != 0)]
    dtypes = file.dtypes # all columns datatype
    dtypes2 = dtypes.loc[(nulls != 0)] # only non null columns datatype

    total = file.isnull().sum().sort_values(ascending=False)
    percent = ((file.isnull().sum()/file.isnull().count()) * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent, dtypes], axis=1, keys=['Total', 'Percent', 'Data Type'])
    if save == 1:
        print(len(nullcols), " missing data, data saves in 'missing_file before_operations.csv'")
        missing_data.to_csv(path+'missing_file before_operations.csv')
    elif save == 2:
        print(len(nullcols), " missing data, data saves in 'missing_file before_operations.csv'")
        missing_data.to_csv(path+'missing_file after_operations.csv')
    else:
        print(len(nullcols), " missing data")

def partial(group, relation):
    pd.set_option('max_columns', None)
    var = relation
    var2 = pd.DataFrame([var[i] for i in group]).T
    var2.to_csv(path + 'partial.csv')
    print("partial relation data has been stored in : partial.csv")

##################################### distribution observation ##############################################

# check hit_map
def hitmap(dataset, target_column, tittle):
    # Complete numerical correlation matrix
    if 0:
        scatter_matrix(dataset,figsize=(20, 15))
        pyplot.show()

    # Partial numerical correlation matrix (target_column)
    corrmat = dataset.corr()
    corr_num = 15 #number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, target_column)[target_column].index
    corr_mat_sales = np.corrcoef(dataset[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
    plt.title(tittle)
    plt.show()


#checking histogram
def hist_plot(dataset, tittle):
    dataset.hist(bins=50, figsize=(20, 15))
    plt.title(tittle)
    plt.show()

#checking skew
def skew_plot(column, tittle):
    plt.figure()
    sns.distplot(column)
    plt.title(tittle)
    plt.show()

#checking outliars
def scatter_plot(file, var, tittle):
    data = pd.concat([file['Outcome'], file[var]], axis=1)
    data.plot.scatter(x=var, y='Outcome', ylim=(0, 1))
    plt.title(tittle)
    plt.show()

# Checking distribution (histogram and normal probability plot)
def general_distribution(file, cell):
    plt.subplot(1, 2, 1)
    sns.distplot(file[file[cell]>0][cell], fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[file[cell]>0][cell], plot=plt)

####################################################################################################
#                                   result functions
####################################################################################################
#Forming a confusion matrix to check our accuracy
def accuracy_calculator(model_name, y_pred, Y_true):
    pp = []
    for p in y_pred:
        if p>0.5:
            pp.append(1)
        else:
            pp.append(0)

    y_pred_f = pp
    cm=confusion_matrix(Y_true,y_pred_f)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
    print('{0} model accuracy : {1:.2f} %'.format(model_name, acc))
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return y_pred_f, acc
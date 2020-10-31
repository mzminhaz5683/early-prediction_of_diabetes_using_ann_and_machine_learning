# coding=utf-8
####################################################################################################
#                                   import
####################################################################################################
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation


import warnings
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings('ignore')

#Limiting floats output to 2 decimal point(s) after dot(.)
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
####################################################################################################
#                                   import local file
####################################################################################################
from programs import checker_v2
from programs import controler
checker_v2.path = path = "./output/Data_observation/"

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
raw_dataset = pd.read_csv('./input/diabetes_raw_1.csv')
####################################################################################################
#                                   data capture
####################################################################################################
train_ID = train['Id'] # tracking train's 'Id'
test_ID = test['Id'] # tracking test's 'Id'

# drop 'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
raw_dataset.drop(['Id'], axis=1, inplace=True)

# remove target column from train
y_train = train.Outcome.reset_index(drop=True) # tracking train's 'Outcome'
df_train = train.drop(['Outcome'], axis = 1) # droping 'Outcome' from train
df_test = test # assign test

all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
####################################################################################################
#                                   data checking
####################################################################################################
print("\n\n~~~~~~~~~~~~~~~~~~~~\n")
print(all_data.dtypes)
print("\n~~~~~~~~~~~~~~~~~~~~\n\n")

# hit_map : 1
if controler.hit_map == 1:
    checker_v2.hitmap(raw_dataset, 'Outcome')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 1
if controler.histogram_show == 1:
    checker_v2.hist_plot(all_data)
    #checker_v2.histogram_show(train)
    #checker_v2.histogram_show(test)

# skew_plot : 1
if controler.check_skw == 1:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn])

# scatter_plot : 1
if controler.check_outliars_numeric_relation == 1:
    for i in train:
        checker_v2.scatter_plot(train, i)
#--------------------------------------------------------------------------------------------------
elif 3 > controler.check_outliars_numeric_relation > 1:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.scatter_plot(train, i)

# missing data checking : 1
if controler.missing_data:
    checker_v2.missing_data(all_data, controler.save_column_name)
####################################################################################################
#                                   shaving all_data as prime
####################################################################################################
if controler.save_all_data:
    all_data.to_csv(path+'all_data_prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data_prime.csv\n')

print('____________________________________________________________________________________')
print('\nall_data shape (Rows, Columns) & Columns-(ID, Outcome): ', all_data.shape)
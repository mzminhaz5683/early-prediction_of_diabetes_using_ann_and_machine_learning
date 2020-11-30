# coding=utf-8
####################################################################################################
#                                   import
####################################################################################################
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation


import warnings
warnings.filterwarnings('ignore')

skew_info = ''
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
#raw_dataset = pd.read_csv('./input/diabetes_raw_1.csv')
####################################################################################################
#                                   data capture
####################################################################################################
train_ID = train.Id.reset_index(drop=True) # tracking train's 'Id'
test_ID = test.Id.reset_index(drop=True) # tracking test's 'Id'
y_train = train.Outcome.reset_index(drop=True) # tracking train's 'Outcome'

test.drop(['Id'], axis=1, inplace=True) # droping 'Id' from test
train.drop(['Id'], axis = 1, inplace=True) # droping 'Id', 'Outcome' from train

raw_dataset = pd.concat([train, test]).reset_index(drop=True) # concatenation
#print(' ~~~~~~~~~~ raw_dataset : ~~~~~~~~~~ \n{0}'.format(raw_dataset.dtypes))
####################################################################################################
#                                   data checking
####################################################################################################
# hit_map : 1
if 0<(controler.hit_map -1+1) or controler.all:
    checker_v2.hitmap(train, 'Outcome', 'train_with_outcome_prime')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 1
if 0<(controler.hist_plot -1+1) or controler.all:
    checker_v2.hist_plot(raw_dataset, 'raw_dataset_Prime')
    #checker_v2.hist_plot(train)
    #checker_v2.hist_plot(test)

# skew_plot : 1
if (0<(controler.skew_plot -1+1) and controler.skew_plot != 4) or controler.all:
    for clmn in raw_dataset:
        checker_v2.skew_plot(raw_dataset[clmn], 'prime')

# scatter_plot : 1
numerics_outliars = train
if 0<(controler.scatter_plot -1+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(numerics_outliars, i, 'prime : '+i)
#--------------------------------------------------------------------------------------------------
elif 0<(controler.scatter_plot -2+1)  or controler.all:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.scatter_plot(train, i, 'prime : '+i)

# missing data checking : 1
if 0<(controler.missing_data -1+1) or controler.all:
    checker_v2.missing_data(raw_dataset, controler.save_column_name)
####################################################################################################
#                                   data dropping
####################################################################################################
train.drop(['Outcome'], axis = 1, inplace=True) # droping 'Id', 'Outcome' from train


all_data = pd.concat([train, test]).reset_index(drop=True) # concatenation
#print('\n ~~~~~~~~~~ all_data : ~~~~~~~~~~ \n{0}'.format(all_data.dtypes))
####################################################################################################
#                                   shaving all_data as prime
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'all_data prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data prime.csv')

print('\nprime :: all_data shape (Rows, Columns) & Columns-(without :: Id, Outcome, classes): ', all_data.shape)
####################################################################################################
#                                   data operation - Multi level missing data handling
####################################################################################################
# single level data handling
#all_data.loc[189, 'SkinThickness'] = 63

if controler.multi_level_Data_Handling  or controler.all:
    ###############~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########################################
    #                               multi-level data handle function    
    #                clmn, p, w = target_column, parameter_column, parameter_weight
    ################~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########################################
    def missing_value_handler(clmn, p, w):
        median = all_data[clmn].median()
        for i in range(0, len(all_data[clmn])):
            if all_data[clmn][i] == 0:
                if len(p) == 3:
                    value = all_data[p[0]][i] * w[0] + all_data[p[1]][i] * w[1] + all_data[p[2]][i] * w[2]
                    all_data[clmn][i] = (median + value)//2
                elif len(p) == 4:
                    value = all_data[p[0]][i] * w[0] + all_data[p[1]][i] * w[1] + all_data[p[2]][i] * w[2] + all_data[p[3]][i] * w[3]
                    all_data[clmn][i] = (median + value)//2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = ['BMI', 'Age', 'Insulin']
    w = [0.22, 0.26, 0.33]
    missing_value_handler('Glucose', p, w)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    p = ['BMI', 'Age', 'SkinThickness']
    w = [0.28, 0.24, 0.21]
    missing_value_handler('BloodPressure', p, w)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    p = ['BMI', 'Insulin', 'BloodPressure']
    w = [0.39, 0.44, 0.21]
    missing_value_handler('SkinThickness', p, w)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    p = ['Glucose', 'BMI', 'SkinThickness']
    w = [0.33, 0.20, 0.44]
    missing_value_handler('Insulin', p, w)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    p = ['Glucose', 'Insulin', 'SkinThickness', 'BloodPressure']
    w = [0.22, 0.20, 0.39, 0.28]
    missing_value_handler('BMI', p, w)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
else:
    print('multi level data handling : off\n')
####################################################################################################
#                                   data operation - Adding new features
####################################################################################################
if controler.class_creating  or controler.all:
    ###########################~~~~~~class generator function~~~~~##################################
    def class_generator(clmn, clmn_target, rng, data_lst):
        j = -1
        for i in all_data[clmn]:
            j += 1
            if i <= rng[0]:
                all_data[clmn_target][j] = data_lst[0]
            elif rng[0] < i <= rng[1]:
                all_data[clmn_target][j] = data_lst[1]
            elif rng[1] < i <= rng[2]:
                all_data[clmn_target][j] = data_lst[2]
            elif rng[2] < i <= rng[3]:
                all_data[clmn_target][j] = data_lst[3]
            else:
                all_data[clmn_target][j] = data_lst[4]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    all_data['AgeClass'] = 0
    class_generator('Age', 'AgeClass', [25, 38, 51, 59], [1,2,3,4,5])

    all_data['GlucoClass'] = 0
    class_generator('Glucose', 'GlucoClass', [60, 80, 140, 180], [2,1,3,4,5])

    all_data['BPClass'] = 0
    class_generator('BloodPressure', 'BPClass', [60, 75, 90, 100], [1,2,3,4,5])

    all_data['BMIClass'] = 0
    class_generator('BMI', 'BMIClass', [18, 25, 30, 40], [1,2,3,4,5])

    all_data['PregClass'] = 0
    class_generator('Pregnancies', 'PregClass', [1, 4, 7, 11], [1,2,3,4,5])
else:
    print('object to numeric converter : deactiveted\n')
####################################################################################################
#                           data checking 2nd time : before transformation
####################################################################################################
# skew_plot : 2
if 0<(controler.skew_plot -2+1)  or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'all_data Before tranformation')

# scatter_plot : 1
if 0<(controler.scatter_plot -3+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(train, i, 'all_data Before tranformation : '+i)

# missing data checking : 2
if 0<(controler.missing_data -2+1)  or controler.all:
    checker_v2.missing_data(all_data, controler.save_column_name + 1)
####################################################################################################
#                                   shaving all_data final
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'all_data before_transformation actual_split.csv')
    print('\nAll prime 2 data has been saved at : '+path+'all_data before_transformation actual_split.csv')

print('\nPrime 2 :: all_data shape (Rows, Columns) & Columns-(without :: Id only): ', all_data.shape)
####################################################################################################
#                                   data operation - transformation
####################################################################################################
#data skewness
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("\n ~~~~~~~~~~ prime skewness ~~~~~~~~~~ ")
print(all_data.skew())
print(" ~~~~~~~~~~ ~~~~~~~~~~~~~~ ~~~~~~~~~~ \n")

if (controler.class_creating == 0 and controler.log_normalization_on_target) or controler.all:
    all_data['SkinThickness']           = np.log1p(all_data['SkinThickness'])  
    all_data['Glucose']                 = np.log1p(all_data['Glucose'])
    all_data['BMI']                     = np.log1p(all_data['BMI'])

    all_data['DiabetesPedigreeFunction']= np.sqrt(np.log1p(np.log1p(all_data['DiabetesPedigreeFunction'])))
    all_data['Insulin']                 = np.sqrt(np.log1p(np.log1p(all_data['Insulin'])))
    all_data['Age']                     = np.sqrt(np.log1p(np.log1p(all_data['Age'])))

elif controler.log_normalization_on_target  or controler.all:

    all_data['SkinThickness']           = np.log1p(all_data['SkinThickness'])
    all_data['PregClass']               = np.log1p(all_data['PregClass'])    
    all_data['Glucose']                 = np.log1p(all_data['Glucose'])
    all_data['BPClass']                 = np.log1p(all_data['BPClass'])
    all_data['BMI']                     = np.log1p(all_data['BMI'])

    all_data['DiabetesPedigreeFunction']= np.sqrt(np.log1p(np.log1p(all_data['DiabetesPedigreeFunction'])))
    all_data['Insulin']                 = np.sqrt(np.log1p(np.log1p(all_data['Insulin'])))
    all_data['Age']                     = np.sqrt(np.log1p(np.log1p(all_data['Age'])))
    all_data['AgeClass']                = np.log1p(np.log1p(all_data['AgeClass']))


    # BMIClass, GlucoClass : are in chipest skew

##############################~~~~~~over fit handinig~~~~~##########################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
overfit = ['Pregnancies']
#overfit = []
for i in all_data.columns:
    counts = all_data[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(all_data)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)
print('dropping overfits : ', overfit)
all_data = all_data.drop(overfit, axis=1).copy()
####################################################################################################
#                                   data checking for 3rd time :: 1st
####################################################################################################
# final data skewness
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("\n ~~~~~~~~~~ final skewness ~~~~~~~~~~ ")
print(all_data.skew())
print(" ~~~~~~~~~~ ~~~~~~~~~~~~~~ ~~~~~~~~~~ \n")

# hist_plot : 2
if 0<(controler.hist_plot -2+1)  or controler.all:
    checker_v2.hist_plot(all_data, 'all_data_Final')
    #checker_v2.hist_plot(final_train)
    #checker_v2.hist_plot(final_test)

# skew_plot : 3
if 0<(controler.skew_plot -3+1) or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'all_data After transformation')

####################################################################################################
#                            all_data spliting & data checking for 3rd time :: 2nd
####################################################################################################
final_train = all_data.iloc[:len(train), :]
final_test = all_data.iloc[len(train):, :]


hitmap_train = final_train.copy()
hitmap_train['Outcome'] = y_train
#print('-------------final_train-------------\n{0}'.format(final_train.dtypes))
#print('-------------hitmap_train-------------\n{0}'.format(hitmap_train.dtypes))

# hit_map : 2
if 0<(controler.hit_map -2+1)  or controler.all:
    checker_v2.hitmap(hitmap_train, 'Outcome', 'train_with_outcome_final')
    #checker_v2.hitmap(train, 'Outcome')

# scatter_plot : 1
if 0<(controler.scatter_plot -4+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(train, i, 'all_data After tranformation : '+i)

####################################################################################################
#                                   shaving all_data final
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'Final_dataset actual_split.csv')
    print('\nAll final data has been saved at : '+path+'Final_dataset actual_split.csv')

print('\nfinal :: all_data shape (Rows, Columns) & Columns-(without :: Id only): ', all_data.shape)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
####################################################################################################
#                                   functions of modeling
####################################################################################################
def get_train_label():
    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \ny_train shape:", y_train.shape)
    return y_train

def get_IDs():
    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \ntest_ID, train_ID shape :", test_ID.shape, train_ID.shape)
    return test_ID, train_ID

def get_train_test_data():
    #print('\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \nX_train dtypes :\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n{0}'.format(final_train.dtypes))
    #print('\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n X_test dtypes :\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n{0}'.format(final_test.dtypes))
    print('\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \nX_train, X_test: ', final_train.shape, final_test.shape)
    return final_train, final_test

    
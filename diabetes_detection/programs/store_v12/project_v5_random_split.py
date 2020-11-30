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

raw_dataset = pd.read_csv('./input/diabetes_raw_1.csv')

# Split dataset into training set and test set
from sklearn.model_selection import train_test_split

Y_dataset = raw_dataset.Outcome.reset_index(drop=True) # tracking train's 'Outcome'
all_data = raw_dataset.drop(['Outcome'], axis = 1) # droping 'Outcome' from dataset
train, test, y_train, y_test = train_test_split(all_data, Y_dataset, test_size=0.2) # 80% training and 20% test
####################################################################################################
#                                   shaving random split
####################################################################################################
random_train = pd.concat([pd.DataFrame(), train])
random_test = pd.concat([pd.DataFrame(), test])
random_y_train = pd.concat([pd.DataFrame(), y_train])
random_y_test = pd.concat([pd.DataFrame(), y_test])

random_train['Outcome'] = random_y_train
random_y_test['Id'] = random_test['Id']

def save_random_split(file_path):
    random_train.to_csv(file_path+'random_train.csv')
    random_test.to_csv(file_path+'random_test.csv')
    random_y_test.to_csv(file_path+'random_y_test.csv')
    print('\n\nsave random_cut')
####################################################################################################
#                                   data capture
####################################################################################################
train_ID = train['Id'] # tracking train's 'Id'
test_ID = test['Id'] # tracking test's 'Id'

# drop 'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
raw_dataset.drop(['Id'], axis=1, inplace=True)

df_train = pd.concat([pd.DataFrame(), train])
df_test  = pd.concat([pd.DataFrame(), test])

all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
####################################################################################################
#                                   data checking
####################################################################################################
# hit_map : 1
if 0<(controler.hit_map -1+1) or controler.all:
    checker_v2.hitmap(raw_dataset, 'Outcome', 'raw_dataset_Prime')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 1
if 0<(controler.hist_plot -1+1) or controler.all:
    checker_v2.hist_plot(all_data, 'All_Data_Prime')
    #checker_v2.hist_plot(train)
    #checker_v2.hist_plot(test)

# skew_plot : 1
if 0<(controler.skew_plot -1+1) or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'prime')

# scatter_plot : 1
numerics_outliars = all_data
if 0<(controler.scatter_plot -1+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(numerics_outliars, i, 'prime : '+i)
#--------------------------------------------------------------------------------------------------
elif 0<(controler.scatter_plot -2+1)  or controler.all:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.scatter_plot(all_data, i, 'prime : '+i)

# missing data checking : 1
if 0<(controler.missing_data -1+1) or controler.all:
    checker_v2.missing_data(all_data, controler.save_column_name)
####################################################################################################
#                                   shaving all_data as prime
####################################################################################################
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'all_data prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data prime.csv\n')

print('__________________________________________________________________________________________')
print('\nprime :: all_data shape (Rows, Columns) & Columns-(without :: Id, Outcome, classes): ', all_data.shape)
####################################################################################################
#                                   data operation - Multi level missing data handling
####################################################################################################
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
else:
    print('object to numeric converter : deactiveted\n')
####################################################################################################
#                           data checking 2nd time : before transformation
####################################################################################################
# skew_plot : 2
if 0<(controler.skew_plot -2+1)  or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'Before tranformation')

# scatter_plot : 1
if 0<(controler.scatter_plot -3+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(all_data, i, 'Before tranformation : '+i)

# missing data checking : 2
if 0<(controler.missing_data -2+1)  or controler.all:
    checker_v2.missing_data(all_data, controler.save_column_name + 1)
####################################################################################################
#                                   data operation - transformation
####################################################################################################
#data skewness
print("\n __________ prime skewness __________ ")
print(all_data.skew())
print(" __________ ______________ __________ \n")

def normalizer(trnsfrm, id):
    if id:
        for clmn in trnsfrm:
            all_data[clmn] = np.log1p(all_data[clmn])
    else:
        for clmn in trnsfrm:
            all_data[clmn] = np.sqrt(all_data[clmn])

if controler.log_normalization_on_target  or controler.all:
    trnsfrm_1 = ['Glucose', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                 'Age', 'AgeClass', 'BPClass']
    normalizer(trnsfrm_1, 1)

    if controler.individual_normalization_show:
        print("\n __________ trnsfrm_1 __________ ")
        print(all_data.skew())
        print(" __________ ______________ __________ \n")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SkinThickness : chipest skewed
    trnsfrm_2 = ['Insulin', 'DiabetesPedigreeFunction',
                 'Age', 'AgeClass']
    normalizer(trnsfrm_2, 1)

    if controler.individual_normalization_show:
        print("\n __________ trnsfrm_2 __________ ")
        print(all_data.skew())
        print(" __________ ______________ __________ \n")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    trnsfrm_3 = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
    normalizer(trnsfrm_3, 0)
    
    if controler.individual_normalization_show:
        print("\n __________ trnsfrm_3 __________ ")
        print(all_data.skew())
        print(" __________ ______________ __________ \n")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Insulin, Age : furthermore transformation gives no significient change

####################################################################################################
#                                   all_data spliting
####################################################################################################

##############################~~~~~~over fit handinig~~~~~##########################################
#overfit = ['Pregnancies']
overfit = []
for i in all_data.columns:
    counts = all_data[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(all_data)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)
print('dropping overfits : ', overfit)
all_data = all_data.drop(overfit, axis=1).copy()
####################################################################################################
#                                   data checking for 3rd time
####################################################################################################
# final data skewness
print("\n __________ final skewness __________ ")
print(all_data.skew())
print(" __________ ______________ __________ \n")
print(skew_info)


final_dataset = all_data
#final_dataset['Outcome'] = Y_dataset # load 'Outcome' in all_data too ... ?

# hit_map : 2
if 0<(controler.hit_map -2+1)  or controler.all:
    checker_v2.hitmap(final_dataset, 'Outcome', 'raw_dataset_2nd_final')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 2
if 0<(controler.hist_plot -2+1)  or controler.all:
    checker_v2.hist_plot(all_data, 'All_data_Final')
    #checker_v2.hist_plot(final_train)
    #checker_v2.hist_plot(final_test)

# skew_plot : 3
if 0<(controler.skew_plot -3+1) or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'After transformation')

# scatter_plot : 1
if 0<(controler.scatter_plot -4+1) or controler.all:
    for i in numerics_outliars:
        checker_v2.scatter_plot(all_data, i, 'After tranformation : '+i)
####################################################################################################
#                                   shaving all_data final
####################################################################################################
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'Final_dataset project_v4 random_split.csv')
    print('\nAll final data has been saved at : '+path+'Final_dataset project_v4 random_split.csv\n')

print('__________________________________________________________________________________________')
print('\nfinal :: all_data shape (Rows, Columns) & Columns-(without :: Id only): ', all_data.shape)
print('\n')
####################################################################################################
#                                   functions of modeling
####################################################################################################
X_train = all_data.iloc[:len(y_train), :]
X_test = all_data.iloc[len(df_train):, :]

def get_train_label():
    print("\n------------------------------\ny_train shape:", y_train.shape)
    return y_train

def get_IDs():
    print("\n------------------------------\ntest_ID, train_ID shape :", test_ID.shape, train_ID.shape)
    return test_ID, train_ID

def get_train_test_data():
    #print('\n------------------------------\nX_train dtypes :\n------------------------------\n{0}'.format(X_train.dtypes))
    #print('\n------------------------------\n X_test dtypes :\n------------------------------\n{0}'.format(X_test.dtypes))
    print('\n------------------------------\nX_train, X_test: ', X_train.shape, X_test.shape)
    return X_train, X_test

def get_actual_result():
    print("\n------------------------------\ny_test shape:", y_test.shape)
    return y_test
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
# hit_map : 1
if controler.hit_map == 1 or controler.all:
    checker_v2.hitmap(raw_dataset, 'Outcome')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 1
if controler.hist_plot == 1 or controler.all:
    checker_v2.hist_plot(all_data)
    #checker_v2.hist_plot(train)
    #checker_v2.hist_plot(test)

# skew_plot : 1
if controler.skew_plot == 1 or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'prime')

# scatter_plot : 1
if controler.scatter_plot == 1 or controler.all:
    for i in train:
        checker_v2.scatter_plot(train, i)
#--------------------------------------------------------------------------------------------------
elif 3 > controler.scatter_plot > 1  or controler.all:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.scatter_plot(train, i)

# missing data checking : 1
if controler.missing_data  or controler.all:
    checker_v2.missing_data(all_data, controler.save_column_name)
####################################################################################################
#                                   shaving all_data as prime
####################################################################################################
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'all_data prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data prime.csv\n')

print('__________________________________________________________________________________________')
print('\nall_data shape (Rows, Columns) & Columns-(ID, Outcome): ', all_data.shape)
####################################################################################################
#                                   data operation - Multi level missing data handling
####################################################################################################
if controler.multi_level_Data_Handling  or controler.all:
    print('multi level data handling : on\n')

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
    class_generator('Glucose', 'GlucoClass', [60, 80, 140, 180], [3,1,2,4,5])

    all_data['BPClass'] = 0
    class_generator('BloodPressure', 'BPClass', [60, 75, 90, 100], [4,2,1,3,5])

    all_data['BMIClass'] = 0
    class_generator('BMI', 'BMIClass', [18, 24, 30, 38], [2,1,3,4,5])
else:
    print('object to numeric converter : deactiveted\n')
####################################################################################################
#                           data checking 2nd time : before transformation
####################################################################################################
# skew_plot : 2
if controler.skew_plot == 2  or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'before tranformation')

# missing data checking : 2
if controler.missing_data  or controler.all:
    checker_v2.missing_data(all_data, controler.save_column_name + 1)
####################################################################################################
#                                   data operation - transformation
####################################################################################################
#data skewness
print("\n __________ prime skewness __________ ")
print(all_data.skew())
print(" __________ ______________ __________ \n")

if controler.log_normalization_on_target  or controler.all:

    if controler.drop_Pregnancies_Glucose  or controler.all:
        # taking out 'Pregnancies' as it has 0 elements,
        # taking out 'Glucose', as it is negatively skewed.
        Pregnancies = all_data['Pregnancies'] # tracking
        Glucose = all_data['Glucose'] # tracking

        # dropping
        all_data.drop(['Pregnancies'], axis=1, inplace=True)
        all_data.drop(['Glucose'], axis=1, inplace=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if 1:
        # Extract numeric variables merged data
        df_merged_num = all_data.select_dtypes(include = ['int64', 'float64'])

        # Make the tranformation of the explanetory variables
        df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew() > 0.5].index])

        # Normal variables
        df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew() < 0.5].index]

        # Merging
        df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis=1)

        #Update numerical variables with transformed variables.
        df_merged_num.update(df_merged_num_all)

        # Creating scaler object.
        scaler = RobustScaler()

        # Fit scaler object on train data.
        scaler.fit(df_merged_num)

        # Apply scaler object to both train and test data.
        df_merged_num_scaled = scaler.transform(df_merged_num)

        # Retrive column names
        df_merged_num_scaled = pd.DataFrame(data = df_merged_num_scaled, columns = df_merged_num.columns, index = df_merged_num.index)

        all_data = df_merged_num_scaled

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 'GlucoClass' : stays at chipest skew at this stage

    # need second transformation
    if controler.transformation_again  or controler.all:
        all_data['Age'] = np.log1p( all_data['Age'])
        all_data['Insulin'] = np.log1p( all_data['Insulin'])

        all_data['AgeClass'] = np.log1p( all_data['AgeClass'])
        all_data['BPClass'] = np.log1p( all_data['BPClass'])

        # negatively skewed now
        all_data['BMIClass'] = np.sqrt( all_data['BMIClass']) # becomes constant at this stage

        # multi layer skew handling
        Glucose = np.log1p(np.sqrt(Glucose))
        all_data['DiabetesPedigreeFunction'] = np.sqrt( np.log1p( all_data['DiabetesPedigreeFunction']))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # concat Pregnancies & Glucose
    if controler.drop_Pregnancies_Glucose  or controler.all:
        all_data['Pregnancies'] = Pregnancies
        all_data['Glucoseg'] = Glucose
        
####################################################################################################
#                                   all_data spliting
####################################################################################################
df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['Outcome'] = y_train

##############################~~~~~~de-couple all_data~~~~~#########################################
y_train = df_train.Outcome.reset_index(drop=True)
final_train = df_train.drop(['Outcome'], axis = 1)
final_test = df_test

##############################~~~~~~over fit handinig~~~~~##########################################
overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(final_train)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)
print('overfit : ', overfit)
final_train = final_train.drop(overfit, axis=1).copy()
final_test = final_test.drop(overfit, axis=1).copy()
print('__________________________________________________________________________________________')
print('\nfinal shape (df_train, y_train, df_test): ',final_train.shape,y_train.shape,final_test.shape)
####################################################################################################
#                                   data checking for 3rd time
####################################################################################################
# final data skewness
print("\n __________ final skewness __________ ")
print(all_data.skew())
print(" __________ ______________ __________ \n")

raw_dataset_2nd = all_data
raw_dataset_2nd['Outcome'] = raw_dataset['Outcome']

# hit_map : 2
if controler.hit_map == 2  or controler.all:
    checker_v2.hitmap(raw_dataset_2nd, 'Outcome')
    #checker_v2.hitmap(train, 'Outcome')

# hist_plot : 2
if controler.hist_plot == 2  or controler.all:
    checker_v2.hist_plot(all_data)
    #checker_v2.hist_plot(final_train)
    #checker_v2.hist_plot(final_test)

# skew_plot : 3
if controler.skew_plot == 3  or controler.all:
    for clmn in all_data:
        checker_v2.skew_plot(all_data[clmn], 'After transformation')
####################################################################################################
#                                   shaving all_data final
####################################################################################################
if controler.save_all_data  or controler.all:
    all_data.to_csv(path+'all_data final.csv')
    print('\nAll prime data has been saved at : '+path+'all_data final.csv\n')

print('__________________________________________________________________________________________')
print('\nall_data shape (Rows, Columns) & Columns-(ID, Outcome): ', all_data.shape)
print('\n\n')
####################################################################################################
#                                   functions of modeling
####################################################################################################
def get_train_label():
    print("y_train of get_train_label():", y_train.shape)
    return y_train

def get_test_ID():
    print("df_test_ID of get_test_ID():", test_ID.shape)
    return test_ID

def get_train_test_data():
    print('Shape of get_train_test_data(): ', final_train.shape, y_train.shape, final_test.shape)
    return final_train, final_test

def project_description(description):
    if controler.file_description:
        description += '~~~~~~~~~~~~~~~~~~~~~~~ Project file data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        file = open(path+'model_description.txt', 'w')
        file.write(description)
        print('__________________________________________________________________________________________')
        print('\nmodel_description has been saved at : '+path+'model_description.txt')
        file.close()
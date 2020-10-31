# coding=utf-8
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation


import warnings
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings('ignore')

#Limiting floats output to 2 decimal point(s) after dot(.)
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
p_description = ''

# import local file
from programs import checker_v2
from programs import controler
checker_v2.path = path = "./output/Data_observation/"

######################################## START ###################################################
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
raw_dataset = pd.read_csv('./input/diabetes_raw_1.csv')

train_ID = train['Id'] # assign
test_ID = test['Id'] # assign

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
raw_dataset.drop(['Id'], axis=1, inplace=True)

################################### Heat Map raw_dataset : 1 ###################################
if controler.hit_map == 1:
    # Complete numerical correlation matrix
    corrmat = raw_dataset.corr()
    f, ax = plt.subplots(figsize=(20, 25))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Partial numerical correlation matrix (salePrice)
    corr_num = 15 #number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, 'Outcome')['Outcome'].index
    corr_mat_sales = np.corrcoef(raw_dataset[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    
    print("\n\n hit_map_raw_dataset 1 : 1------\n\n")
    plt.show()
######################################## 2. Out-liars Handling ######################################

################################# numerical relationship : 1 #######################################
if controler.check_outliars_numeric_relation == 1:
    for i in train:
        checker_v2.numerical_relationship(train, i)

elif 3 > controler.check_outliars_numeric_relation > 1:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.numerical_relationship(train, i)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
############################################# histogram : 1 ####################################
if controler.histogram_show == 1:
    checker_v2.histogram_show(raw_dataset)
    checker_v2.histogram_show(train)
    checker_v2.histogram_show(test)
########################## concatenation of train & test ################################
y_train = train.Outcome.reset_index(drop=True) # assign
df_train = train.drop(['Outcome'], axis = 1) # drop
df_test = test # assign

all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
#--------------------------------------------------------------------------------------------------
################################### Save data ###################################################
if controler.save_all_data:
    all_data.to_csv(path+'all_data_prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data_prime.csv\n')
    p_description +="\n\nAll prime data has been saved at : '+path+'all_data_prime.csv\n"

print('____________________________________________________________________________________')
print('\nall_data shape (Rows, Columns) & Columns-(ID, Outcome): ', all_data.shape)


####################################### data operation #######################################
#----------------------------------- Missing data checking ---------------------------------------
if controler.missing_data:
    checker_v2.missing_data(all_data, controler.save_column_name) # checking missing data 1/0 for save as file or not

#~~~~~~~~~~~~~~~~~~~~~~~~ Multi-level (Data Handling) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if controler.multi_level_Data_Handling :
    print('multi level data handling : 1\n')
    p_description +='\multi level data handling : 1\n'

    #####################################################################################
    # multi-level data handle function    
    #target_column, parameter_column, weight
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
    #####################################################################################

    p = ['BMI', 'Age', 'Insulin']
    w = [0.22, 0.26, 0.33]
    missing_value_handler('Glucose', p, w)
    #####################################################################################
    
    p = ['BMI', 'Age', 'SkinThickness']
    w = [0.28, 0.24, 0.21]
    missing_value_handler('BloodPressure', p, w)
    #####################################################################################

    p = ['BMI', 'Insulin', 'BloodPressure']
    w = [0.39, 0.44, 0.21]
    missing_value_handler('SkinThickness', p, w)
    #####################################################################################
        
    p = ['Glucose', 'BMI', 'SkinThickness']
    w = [0.33, 0.20, 0.44]
    missing_value_handler('Insulin', p, w)
    #####################################################################################
            
    p = ['Glucose', 'Insulin', 'SkinThickness', 'BloodPressure']
    w = [0.22, 0.20, 0.39, 0.28]
    missing_value_handler('BMI', p, w)
    #####################################################################################
    #------------------------------------------------------------------------------------
else:
    print('multi level data handling : 0\n')
    p_description +='\multi level data handling : 0\n'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ missing data ccheck again~~~~~~~~~~~~~~~~~~~~~~
if controler.missing_data:
    checker_v2.missing_data(all_data, controler.save_column_name + 1) # checking missing data 1/0 for save as file or not
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
############################## adding new feature ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ class conversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if controler.class_conversion:
    #class creation

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('for class_generator')
    all_data['AgeClass'] = 0
    class_generator('Age', 'AgeClass', [25, 38, 51, 59], [1,2,3,4,5])

    all_data['GlucoClass'] = 0
    class_generator('Glucose', 'GlucoClass', [60, 80, 140, 180], [4,2,1,3,5])

    all_data['BPClass'] = 0
    class_generator('BloodPressure', 'BPClass', [60, 75, 90, 100], [4,2,1,3,5])

    all_data['BMIClass'] = 0
    class_generator('BMI', 'BMIClass', [18, 24, 30, 38], [2,1,3,4,5])
else:
    print('object to numeric converter : deactiveted\n')
    p_description +='\nobject to numeric converter : deactiveted\n'

############################### saving data ###################################################

if controler.save_column_name:
    # get only column names and transposes(T) row into columns
    cName_all_data = all_data.head(0).T
    cName_all_data.to_csv(path+'save_column_names.csv')
    print('Columns names saved at :'+path+'save_column_names.csv')
    p_description +='\nColumns names saved at :'+path+'save_column_names.csv'

if controler.save_all_data:
    all_data.to_csv(path+'all_data_secondary.csv')
    print('\nAll modified data has been saved at : '+path+'all_data_secondary.csv')
    p_description +='\n\nAll modified data has been saved at : '+path+'all_data_secondary.csv'

####################### skewed & log1p on numerical features ###########################################
if controler.check_skw:
    for i in all_data:
        checker_v2.skewness(i)

if controler.skw_log:
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

########################################  Heat Map : 2 ##############################################
hit_map_raw_dataset = all_data
hit_map_raw_dataset['Outcome'] = raw_dataset['Outcome']

if controler.hit_map == 2:
    # Complete numerical correlation matrix
    corrmat = hit_map_raw_dataset.corr()
    f, ax = plt.subplots(figsize=(20, 13))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Partial numerical correlation matrix (Outcome)
    corr_num = 15  # number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, 'Outcome')['Outcome'].index
    corr_mat_sales = np.corrcoef(hit_map_raw_dataset[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    print("hit_map 2 : 1")
    p_description +="\nhit_map 2 : 1"
    plt.show()

##################################### numerical relationship ################################
if controler.check_outliars_numeric_relation == 3:
    numerics_outliars = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    for i in numerics_outliars:
        checker_v2.numerical_relationship(train, i)

##################################### data split #################################################

df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['Outcome'] = y_train

#################################### de-couple all_data ####################################
y_train = df_train.Outcome.reset_index(drop=True)
final_train = df_train.drop(['Outcome'], axis = 1)
final_test = df_test

####################################### over fit handinig ####################################
overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(final_train)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)

print('overfit : ', overfit)

############################ final shape #####################################################
final_train = final_train.drop(overfit, axis=1).copy()
final_test = final_test.drop(overfit, axis=1).copy()

print('final shape (df_train, y_train, df_test): ',final_train.shape,y_train.shape,final_test.shape)

######################################### histogram : 2 #######################################
if controler.histogram_show == 2:
    checker_v2.histogram_show(hit_map_raw_dataset)
    checker_v2.histogram_show(final_train)
    checker_v2.histogram_show(final_test)

def get_train_label():
    print("y_train of get_train_label():", y_train.shape)
    global p_description
    p_description +="\ny_train of get_train_label():", y_train.shape
    return y_train

def get_test_ID():
    print("df_test_ID of get_test_ID():", test_ID.shape)
    global p_description
    p_description +="\ndf_test_ID of get_test_ID():", test_ID.shape
    return test_ID

def get_train_test_data():
    print('Shape of get_train_test_data(): ', final_train.shape, y_train.shape, final_test.shape)
    global p_description
    p_description +='\nShape of get_train_test_data(): ', final_train.shape, y_train.shape, final_test.shape
    return final_train, final_test

def project_description(description):
    if controler.file_description:
        description += '~~~~~~~~~~~~~~~~~~~~~~~ Project file data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        global p_description
        description += p_description
        file = open(path+'process_description.txt', controler.file_open_order)
        file.write(description)
        print('\n____________________________________________________________________________________')
        p_description +='\n\n____________________________________________________________________________________'
        print('process_description has been saved at : '+path+'process_description.txt')
        p_description +='\nprocess_description has been saved at : '+path+'process_description.txt'
        file.close()

    
if controler.local_project_description:
    project_description('')

# coding=utf-8
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation


import warnings

from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

#Limiting floats output to 1 decimal point(s) after dot(.)
pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x))
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
p_description = ''

# import local file
from programs import checker_v2
from programs import controler
checker_v2.path = path = "./output/Data_observation/"
######################################## design #(40) + string + #(*) = 100 ######################




######################################## START ###################################################'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw : 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

train_ID = train['Id'] # assign
test_ID = test['Id'] # assign

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
print("\n\n------comlple loading csv------\n\n")


######################################## Heat Map : 1 ################################################
if controler.hit_map:
    # Complete numerical correlation matrix
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(20, 25))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Partial numerical correlation matrix (salePrice)
    corr_num = 15 #number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, 'Outcome')['Outcome'].index
    corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    
    print("\n\n hit_map 1 : 1------\n\n")
    p_description +="\n\n\n hit_map 1 : 1------\n\n"
    plt.show()


######################################## 1. Data Handling #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if controler.log_normalization_on_target: # no need for 0,1 range
    print("Outcome_log1p : on\n")
    p_description +="\nOutcome_log1p : on\n"
    train['Outcome'] = np.log1p(train['Outcome'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######################################## 2. Out-liars Handling #########################################
#..............................2a numerical analyzing...................................

p_description += '---------------- Numerical_outliars : Top ---------------------\n'
if controler.check_outliars_numeric:
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in train.columns:
        if train[i].dtype in numeric_dtypes:
            numerics.append(i)

    if controler.save_column_name:
        # get only column names and transposes(T) row into columns
        numeric_data = train[numerics]
        cName_n_data = numeric_data.head(0).T
        cName_n_data.to_csv(path + 'numeric_save_column_names.csv')
        print('Numeric columns names saved at :' + path + 'numeric_save_column_names.csv')

    if 1:
        for i in numerics:
            checker_v2.numerical_relationship(train, i)
    else:
        numerics_outliars = ['', '', '']
        for i in numerics_outliars:
            checker_v2.numerical_relationship(train, i)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
drop_index = train[(train[''] > 3000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train[''] > 3000)].index\n"
'''
#####################################################################################################
#...............................2b categorical analyzing.................................
if controler.check_outliars_objects:
    objects = []
    for i in train.columns:
        if train[i].dtype == object:
            objects.append(i)

    if controler.save_column_name:
        # get only column names and transposes(T) row into columns
        object_data = train[objects]
        cName_n_data = object_data.head(0).T
        cName_n_data.to_csv(path + 'object_save_column_names.csv')
        print('Object columns names saved at :' + path + 'Object_save_column_names.csv')

    if 1:
        for i in objects:
            checker_v2.categorical_relationship(train, i)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
########################## concatenation of train & test ################################
y_train = train.Outcome.reset_index(drop=True) # assign
df_train = train.drop(['Outcome'], axis = 1) # drop
df_test = test # assign


all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
#dtypes = all_data.dtypes
if controler.save_all_data:
    all_data.to_csv(path+'all_data_prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data_prime.csv\n')
    p_description +="\n\nAll prime data has been saved at : '+path+'all_data_prime.csv\n"


print('____________________________________________________________________________________')
p_description +='\n____________________________________________________________________________________'
print('\nall_data shape (Rows, Columns) & Columns-(ID, Outcome): ', all_data.shape)



####################################### data operation #######################################
if controler.missing_data:
    checker_v2.missing_data(all_data, 1) # checking missing data 1/0 for save as file or not


#......................... single level (Data Handling) .................................
if controler.single_level_Data_Handling :
    print('single level data handling : 1\n')
    p_description +='\nsingle level data handling : 1\n'
else:
    print('single level data handling : 0\n')
    p_description +='\nsingle level data handling : 0\n'

#~~~~~~~~~~~~~~~~~~~~~~~~ Multi-level (Data Handling) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if controler.multi_level_Data_Handling :
    print('multi level data handling : 0\n')
    p_description +='\multi level data handling : 0\n'
    # 'NA' means Special value
    all_data[''] = all_data[''].fillna('')

    #------------------------------------------------------------------------------------
    # categorical( = need to be numerical)
    c2n = []
    #'NA' means most frequest value
    common_vars = [ ]
    common_vars += c2n
    for var in common_vars:
        all_data[var] = all_data[var].fillna(all_data[var].mode()[0])
    p_description += "common_vars += c2n &\n'', '' = fillna(all_data[var].mode()[0])\n"
    #------------------------------------------------------------------------------------
    # categorical( = need to be numerical)
    common_vars = []
    # categorical 'NA' means 'None'
    for col in common_vars:
        all_data[col] = all_data[col].fillna('None')
    #------------------------------------------------------------------------------------
    # numerical 'NA' means 0
    common_vars = ['', '', '', '', '']
    for col in common_vars:
        all_data[col] = all_data[col].fillna(0)
    p_description += "'', '' = fillna(0)\n"
    #------------------------------------------------------------------------------------
    # 'NA'means most or recent common value according to (base on) other special groups
    all_data[''] = all_data.groupby('')[''].transform(lambda x: x.fillna(x.mode()[0]))
    #------------------------------------------------------------------------------------
    # condition of data description
    all_data[''] = all_data[''].fillna(all_data[''])
    #------------------------------------------------------------------------------------
    # Collecting all object type feature and handling multi level null values
    objects = []
    for i in all_data.columns:
        if all_data[i].dtype == object:
            objects.append(i)

    all_data.update(all_data[objects].fillna('None'))
    #------------------------------------------------------------------------------------
    # Collectting all numeric type feature and handling multi level null values
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in all_data.columns:
        if all_data[i].dtype in numeric_dtypes:
            numerics.append(i)

    all_data.update(all_data[numerics].fillna(0))
else:
    print('multi level data handling : 0\n')
    p_description +='\multi level data handling : 0\n'

####################### conversion of data-type ####################################
# (categorical) converting numerical variables that are actually categorical
if controler.objective_conversion == 1:
    cols = ['', '', '', '', '']
    for var in cols:
        all_data[var] = all_data[var].astype(str)
    print("cols = {0} =  str\n".format(cols))
    p_description +="\ncols = {0} =  str\n".format(cols)
#---------------------------------------------------------------------------------

print('skew removed\n')
p_description +='\nskew removed\n'
if 0:
    all_data.to_csv(path+'all_data_secondary.csv')
    print('\nAll modified data has been saved at : '+path+'all_data_secondary.csv')
    p_description +='\n\nAll modified data has been saved at : '+path+'all_data_secondary.csv'

# dropping the columns which have a large amount of distance between it's value used amount
drop_columns = []
all_data = all_data.drop([i for i in drop_columns], axis=1)
print("\ndrop_columns = {0}\n".format(drop_columns))
p_description +="\n\ndrop_columns = {0}\n".format(drop_columns)

############################## adding new feature ######################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if controler.missing_data:
    checker_v2.missing_data(all_data, 0) # checking missing data 1/0 for save as file or not
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

####################### skewed & log1p on numerical features ###########################################

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
# Pass the index of index df_merged_num, otherwise it will sum up the index.


######################### categorical ################################################

"""Let's extract categorical variables first and convert them into category."""
df_merged_cat = all_data.select_dtypes(include = ['object']).astype('category')


if controler.o2n_converter:
    print('object to numeric converter : 1\n')
    p_description +="\nobject to numeric converter : 1\n"
    # !!!!!!!!!!!!!!!!!!!!!!!!!  fillna(0)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # creating a set of all categorical(Ordinal) variables with a specific value to the characters
    dic = {'Grvl': 3, 'Pave': 6, 'NA': 0, 'None' : 0}
    df_merged_cat['Alley'] = checker_v2.data_converter(dic, df_merged_cat, 'Alley')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['FireplaceQu'] = checker_v2.data_converter(dic, df_merged_cat, 'FireplaceQu')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageQual'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtQual'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageCond'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageCond')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtCond'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtCond')

    dic = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageFinish'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageFinish')


    '''Extract label encoded variables'''
    df_merged_label_encoded = df_merged_cat.select_dtypes(include=['int64'])


    '''Finally join processed categorical and numerical variables'''
    all_data = pd.concat([df_merged_num_scaled, df_merged_label_encoded], axis=1)
else:
    print('object to numeric converter : deactiveted\n')
    p_description +='\nobject to numeric converter : deactiveted\n'


if controler.save_column_name:
    # get only column names and transposes(T) row into columns
    cName_all_data = all_data.head(0).T
    cName_all_data.to_csv(path+'save_column_names.csv')
    print('Columns names saved at :'+path+'save_column_names.csv')
    p_description +='\nColumns names saved at :'+path+'save_column_names.csv'


df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['Outcome'] = y_train



########################################  Heat Map : 2 ##############################################
if controler.hit_map:
    # Complete numerical correlation matrix
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(20, 13))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Partial numerical correlation matrix (salePrice)
    corr_num = 15  # number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
    corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    print("hit_map 2 : 1")
    p_description +="\nhit_map 2 : 1"
    plt.show()
#________________________________________________________________________________________________


#################################### creating dummy & de-couple all_data #########################
y_train = df_train.Outcome.reset_index(drop=True)
final_train = df_train.drop(['Outcome'], axis = 1)
final_test = df_test

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

print('final shape (df_train, y_train, df_test): ',final_train.shape,y_train.shape,final_test.shape)

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
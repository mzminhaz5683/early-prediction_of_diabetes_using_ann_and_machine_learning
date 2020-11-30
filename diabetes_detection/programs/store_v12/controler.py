all = 0 #0/1
####################################################################################################
#                                   file handling
####################################################################################################
file_description = 0 #0/1
save_column_name = 0 #0/1
save_all_data = 0 #0/1
####################################################################################################
#                                   data handling
####################################################################################################
class_creating = 1 # 0/1
multi_level_Data_Handling = 1 #0/1
####################################################################################################
#                                   data checkig
####################################################################################################
hit_map = 0 #0/1/2
hist_plot = 0 #0/1/2
skew_plot = 0 #0/1/2/3 /4 for seeing the transformation effect only
scatter_plot = 0 #0/1
missing_data = 0 #0/1/2
####################################################################################################
#                                   data transformation
####################################################################################################
log_normalization_on_target = 1 #0/1
individual_normalization_show = 0 #0/1
####################################################################################################
#                                   model controller
####################################################################################################
rndm_state = 42         # 0 best: 10
n_estimators = 10       # 10 best: 15
criterion = 'gini'      # entropy , gini best: gini
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
project_version = 3     # 3/5
resut_file_name = 'model'
####################################################################################################
#                                  ANN model controller
####################################################################################################
# saved_model_dir = './output/checkpoint/saved/{0}.h5'.format(test_parameters)
test_parameters = '87.7_ann_model'

activate_train = 0 #0/1
target_acc = 80
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
initial_weight = 0.01
alpha_lrelu = 0 # 0/0.1
leakyRelu = 1 #0/1
train_epochs = 300

dropout = [0.10, 0.15, 0.20]
#dns = [32, 32, 64, 128, 128, 256]
dns = [32, 64, 64, 128, 128, 128, 256]
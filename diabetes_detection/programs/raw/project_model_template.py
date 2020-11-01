import pandas as pd
####################################################################################################
#                                   import local
####################################################################################################
# import local files & performance parameters
# noinspection PyBroadException
try:
    f_counter = open('./output/contents/model_counter.txt', 'r')
    counter = int(f_counter.read())
    f_counter.close()
except:
    counter = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f_counter = open('./output/contents/model_counter.txt', 'w')
counter = str(counter+1)
f_counter.write(counter)
f_counter.close()
####################################################################################################
#                                   import documents
####################################################################################################
submission = pd.read_csv("./input/sample_submission.csv")
from programs import project_v2 as project_analyser
output = 'project_v2'
file_formate = '_m2_'+counter+'.csv'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_train, X_test = project_analyser.get_train_test_data()
y = y_train = project_analyser.get_train_label()
####################################################################################################
#                                   Model Start
####################################################################################################

















####################################################################################################
#                                   save result
####################################################################################################
submission.to_csv("./output/"+output+file_formate, index=False)
print('\n\n\n________________Stage finished : {0}___________________'.format(counter))
print('\n\nSubmissin sucessfull saved in output with the name')
print(' ~>> '+output+file_formate)
print('\n\n__________________________________________________________')
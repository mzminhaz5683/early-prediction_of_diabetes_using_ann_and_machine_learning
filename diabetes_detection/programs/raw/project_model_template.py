import pandas as pd
####################################################################################################
#                                   import local
####################################################################################################
# import local files & performance parameters
try:
    f_counter = open('./output/model_counter.txt', 'r')
    counter = int(f_counter.read())
    f_counter.close()
except:
    counter = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f_counter = open('./output/model_counter.txt', 'w')
counter = str(counter+1)
f_counter.write(counter)
f_counter.close()
####################################################################################################
#                                   import documents
####################################################################################################
submission = pd.read_csv("./input/sample_submission.csv")
from programs import project_v2 as project_analyser

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output = 'text'
file_formate = '_model_'+counter+'.csv'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_train, X_test = project_analyser.get_train_test_data()
y = y_train = project_analyser.get_train_label()
####################################################################################################
#                                   Model Start
####################################################################################################



####################################################################################################
#                                   save result
####################################################################################################
submission.to_csv("./output/submission/"+output+file_formate, index=False)
print('\n________________Stage finished for count: {0}___________________\n'.format(counter))
print('Submissin sucessfull saved in output with the name')
print(' ~>> '+output+file_formate)
print('\n\n')
import os

path = "output/set_of_+80_acc/{0:.2f}_set".format(12.5432)

try:
    os.mkdir(path)
    print('Path created')
except:
    print('Can not')
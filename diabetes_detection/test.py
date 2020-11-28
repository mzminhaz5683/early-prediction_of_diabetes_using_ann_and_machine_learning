lst1=[1,0,1,1,0]
lst2=[0,0,0,1,1]
lst3=[1,0,0,1,1]
lst4=[0,1,1,0,1]


lst_combo = [lst1, lst2, lst3, lst4]

import numpy as np
from statistics import mode

result = np.array([])

for i in range(0, len(lst_combo[0])):
    try:
        result = np.append(result, mode([clm[i] for clm in lst_combo]))
    except:
        result = np.append(result, lst_combo[0][i])

print(lst1, len(lst1))
print(result, len(result))
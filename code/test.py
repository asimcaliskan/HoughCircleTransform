import numpy as np

asd = np.full((3,1),fill_value=0,dtype=np.uint64)
asd[0, 0] = 1
asd[1, 0] = 2
asd[2, 0] = 4
print(asd)
asd[asd < 20] = 0
print(asd)
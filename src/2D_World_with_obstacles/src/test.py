import numpy as np
from time import time
start = time()
x1 = 0.0
x2 = 1.0
y1 = 0.0
y2 = 2.0
x = np.array([y2-y1,x2-x1])
angle = np.arctan2(x[0],x[1]) - 90 * np.pi / 180
end = time()
print('time:',(end-start)*1000)

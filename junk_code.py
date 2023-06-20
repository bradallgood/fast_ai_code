# Write Python3 code here
import timeit
import time
import random
import numpy as np
import matplotlib.pyplot as plt


#print(timeit.timeit(stmt='x=0',setup='',timer=time.perf_counter,number=1,globals=None))



print(timeit.timeit(stmt='x=0',setup='',number=1000000))

print(timeit.timeit(stmt='while x < 1000000: x +=1',setup='x=0',number=1))

print(timeit.timeit(stmt='for x in range(1000000): pass',setup='',number=1))


time_list = timeit.repeat(stmt='for x in range(1000000): pass',setup='',number=1,repeat=5)

print(f'{time_list}')

x_lables = np.arange(len(time_list))
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,8))

ax1.plot(x_lables,time_list)
ax2.bar(x_lables,time_list)
plt.show()



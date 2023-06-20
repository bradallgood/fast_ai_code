# Write Python3 code here
import timeit
import numpy as np
import matplotlib.pyplot as plt


def WhileRun(random_normal):
    print("in the function WhileRun")
    abc = 0
    while_array = []
    print(f'random normal is {random_normal}')  
    while abc < len(random_normal):
        while_array.append(random_normal[abc] + 1)
        abc += 1
    return while_array

def ForRun(random_normal):
    for_array = []
    for cbd in random_normal:
        for_array.append(cbd + 1)
    return for_array

def Add(xx):
    return xx + 1

repeat = 5
numbers = 1000000

random_normal = np.random.normal(0, 1, numbers)
while_out = []
for_our = []

print("in while")
while_list = timeit.repeat(stmt='WhileRun(random_normal)',setup='',number=1,repeat=repeat,globals=globals())

print("in for")
for_list = timeit.repeat(stmt='ForRun(random_normal)',setup='',number=1,repeat=repeat,globals=globals())

print("in map")
map_list = timeit.repeat(stmt='map_out = map(Add,random_normal)',setup='',number=1,repeat=repeat,globals=globals())

#print(list(map_list))
x_lables = np.arange(len(while_list))

#fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(12,6),sharex=True,sharey=True)

fmt = '%.7f'
label_type = 'center'

fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(12,10),sharex=False,sharey=False)
fig.patch.set_facecolor('grey')
fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
fmt = '%.7f'
label_type = 'center'
color1 = "#8A5AC2"
color2 = "#3575D5"

fig.suptitle('Testit scenario testing')
ax1.set_title('Using while')
ax1_container = ax1.bar(x_lables,while_list)
ax1.bar_label(ax1_container,fmt=fmt,label_type=label_type)

ax2.set_facecolor('yellow')
ax2.grid(visible = True)
ax2.set_title('Using for')
ax2_container = ax2.bar(x_lables,for_list,color = get_color_gradient(color1, color2, len(x_lables)))
ax2.bar_label(ax2_container,fmt=fmt,label_type=label_type)

ax3.set_title('Using map')
ax3_container = ax3.bar(x_lables,map_list)
ax3.bar_label(ax3_container,fmt=fmt,label_type=label_type)

plt.show()



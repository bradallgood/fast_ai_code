import matplotlib.pyplot as plt
import numpy as np
import timeit
from statistics import mean

def annotate_axes(ax, text, fontsize):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="red")
    
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

def make_chart(k,list,x_l,fontsize=10):
    axd[k].grid(visible = True)
    container = axd[k].bar(x_l,list,)
    axd[k].bar_label(container,fmt=fmt,label_type=label_type,fontsize=6)
    axd[k].set_title(k, fontsize=fontsize, color="black")

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
repeat = 5
numbers = 1000000

random_normal = np.random.normal(0, 1, numbers)

while_list = timeit.repeat(stmt='WhileRun(random_normal)',setup='',number=1,repeat=repeat,globals=globals())
for_list = timeit.repeat(stmt='ForRun(random_normal)',setup='',number=1,repeat=repeat,globals=globals())
map_list = timeit.repeat(stmt='map_out = map(Add,random_normal)',setup='',number=1,repeat=repeat,globals=globals())
x_lables = np.arange(len(while_list))

while_mean = mean(while_list)
for_mean = mean(for_list)
map_mean = mean(map_list)

iter_list = [while_mean,for_mean,map_mean]
iter_label = ['While','For','Map']

fmt = '%.7f'
label_type = 'center'
fig, axd = plt.subplot_mosaic([['compare', 'compare','compare'],
                               ['while_chart', 'for_chart','map_chart']],
                              figsize=(10, 5), layout="constrained")
fig.patch.set_facecolor('grey')
make_chart('compare',iter_list,iter_label)
make_chart('map_chart',map_list,x_lables)
make_chart('while_chart',while_list,x_lables)
make_chart('for_chart',for_list,x_lables)
fig.suptitle('Compare While/For/Map')
plt.show()

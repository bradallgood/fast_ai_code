import matplotlib as plt
from fastai.vision.all import *



def f(x):
    return x**2

def slope(x)

def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, step=100, figsize=(12,8)):
    x = torch.linspace(min,max,step)
    print(x)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)
    plt.grid(visible=True)
    lx = torch.linspace(-2,2,100)
    ly = 4*lx + -1.5
    ax.plot(lx,ly)

plot_function(f)
plt.show()


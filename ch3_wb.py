from ipywidgets import interact
from fastai.basics import *
import numpy as np

plt.rc('figure', dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
    x = torch.linspace(min,max, 100)[:,None]
    if ylim: plt.ylim(ylim)
    plt.plot(x, f(x), color)
    if title is not None: plt.title(title)
    plt.show()
    plt.close()

def f(x): return 3*x**2 + 2*x + 1

def f1(x): return 3*x + 1

def quad(a, b, c, x): return a*x**2 + b*x + c

def mk_quad(a,b,c): return partial(quad, a,b,c)

def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)

def add_noise(x, mult, add): 
    #print(f'x is: {x}')
    #print(f'with noise: {x * (1+noise(x,mult)) + noise(x,add)}')
    return x * (1+noise(x,mult)) + noise(x,add)

def plot_quad(a, b, c):
    plt.scatter(x,y)
    loss = mae(f(x), y)
    plot_function(f, ylim=(-3,12), title=f"MAE: {loss:.2f}")
    return loss

def mae(preds, acts): return (torch.abs(preds-acts)).mean()

def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)

def make_chart(k,list,x_l,fontsize=10):
    axd[k].grid(visible = True)
    container = axd[k].bar(x_l,list,)
    axd[k].bar_label(container,fmt=fmt,label_type=label_type,fontsize=6)
    axd[k].set_title(k, fontsize=fontsize, color="black")

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

np.random.seed(42)
x = torch.linspace(-5, 5, steps=1000)[:,None]
y = add_noise(f(x), 0.05, 1.0)
abc = torch.tensor([1.1,1.1,1.1]).requires_grad_()

for i in range(100):
    print(f'--------------- Round {i} -------------------------')

    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.0001

    if i == 0:
        loss_hold = np.array(loss.clone().detach().numpy())
        weight_hold = np.array([abc.clone().detach().numpy()])
        grad_hold = np.array([abc.grad.clone().detach().numpy()])
    else:
        loss_hold = np.append(loss_hold,loss.clone().detach().numpy())    
        weight_hold = np.append(weight_hold,[abc.clone().detach().numpy()],axis=0)
        grad_hold = np.append(grad_hold,[abc.grad.clone().detach().numpy()],axis=0)

x_hold = []
for w in range(len(loss_hold)):
    x_hold.append(w)

# ------------------------------------------ chart code -------------------------------
fmt = '%.7f'
label_type = 'center'
fig, axd = plt.subplot_mosaic([['loss', 'loss','loss'],
                               ['aweight','bweight','cweight'],
                               ['agrad','bgrad','cgrad']],
                              figsize=(10, 5), layout="constrained")
fig.patch.set_facecolor('grey')
make_chart('loss',loss_hold,x_hold)
make_chart('aweight',weight_hold[:,0],x_hold)
make_chart('bweight',weight_hold[:,1],x_hold)
make_chart('cweight',weight_hold[:,2],x_hold)
make_chart('agrad',grad_hold[:,0],x_hold)
make_chart('bgrad',grad_hold[:,1],x_hold)
make_chart('cgrad',grad_hold[:,2],x_hold)
fig.suptitle('Loss / Weights / Grad')
plt.show()
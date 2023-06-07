import os
from fastai.vision.all import *
from datetime import datetime

matplotlib.rc('image', cmap='Greys')

#Get current time
def time_getter():
    return str(datetime.now())

# ----------------------------------------------------------
# Set the basedir for the handwritten number samples
#-----------------------------------------------------------
base_dir = 'c:\\users\\brada\\onedrive\Desktop\\mnist_sample'
path = Path(base_dir)

# ----------------------------------------------------------
# 1) load the files associated with 3's and 7's into a list
# 2) use the list to generate tensors for each image
# 3) Adjust the values for the tensors to be 0-1 instead of 0-255
#-----------------------------------------------------------
def prep_data():

   print("1) Making a list of training files to be used for 3's and 7's               " + time_getter() )
   threes = (path/'train'/'3').ls().sorted()
   sevens = (path/'train'/'7').ls().sorted()

   print("2) Building tensors for 3's and 7's training files                          " + time_getter())
   seven_tensors = [tensor(Image.open(o)) for o in sevens]
   three_tensors = [tensor(Image.open(o)) for o in threes]

   print("3) Stacking and adjusting tensor values for 3's and 7's training files      " + time_getter())
   stacked_sevens = torch.stack(seven_tensors).float()/255
   stacked_threes = torch.stack(three_tensors).float()/255

   print("4) Calculating mean for all pixels in the training images for 3 and 7       " + time_getter())   
   mean3 = stacked_threes.mean(0)
   mean7 = stacked_sevens.mean(0)

   print("5) Stacking and adjusting tensor values for 3's and 7's validation files    " + time_getter())
   valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
   valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
   print("6) Adjusting tensor values for 3's and 7's validation files                 " + time_getter())
   valid_3_tens = valid_3_tens.float()/255
   valid_7_tens = valid_7_tens.float()/255

   print("Done                                                                        " + time_getter())
#----------------------------------------------------------------------------------------------------------
# Working area 
#---------------------------------------------------------------------------------------------------------
def mse(preds, targets): return ((preds-targets)**2).mean()

def f1(x): return (x**2).sum()
def f(t, params):
    a,b,c = params
    #print('function f values: ' + str(params))
    return a*(t**2) + (b*t) + c
    
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds


time = torch.arange(0,20).float()
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
params = torch.randn(3).requires_grad_()
orig_params = params.clone()
lr = 1e-5

for i in range(10):
   print('\n==== Round ' + str(i) + ' =========')
   preds = f(time,params)
   loss = mse(preds,speed)
   print('loss:  ' + str(loss))
   loss.backward()
   print('Before Params : ' + str(params.data))
   print('Params Grads  : ' + str(params.grad))
   params.data -= lr * params.grad.data
   print('After Params : ' + str(params.data))
   params.grad = None
   print('After .grad= None : ' + str(params.data))


''' simple with single value
xt = tensor(3.).requires_grad_()
print('xt:      ' + str(xt))

yt = f1(xt)
print('yt:      ' + str(yt))

yt.backward()
print('xt.grad:      ' + str(xt.grad))

print('Before xt.data: ' + str(xt.data))
xt.data -= 0.1 * xt.grad.data
print('After xt.data:  ' + str(xt.data))

print('Plain old xt: ' + str(xt))
'''

'''  simple with multiple values
xt = tensor(3.,4.,10.).requires_grad_()
print('xt:      ' + str(xt))

yt = f1(xt)
print('yt:      ' + str(yt))

yt.backward()
print('xt.grad:      ' + str(xt.grad))

print('Before xt.data: ' + str(xt.data))
xt.data -= 0.1 * xt.grad.data
print('After xt.data:  ' + str(xt.data))

print('Plain old xt: ' + str(xt))

'''
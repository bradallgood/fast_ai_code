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


#---------------------------------------------------------------
# working area
#---------------------------------------------------------------
prep_data()

# We already have our independent variables x
# —these are the images themselves. 
# We'll concatenate them all into a single tensor, 
# and also change them from a list of matrices (a rank-3 tensor) to a list of vectors (a rank-2 tensor). 
# We can do this using view, which is a PyTorch method that changes the shape of a tensor without changing its contents. 
# -1 is a special parameter to view that means "make this axis as big as necessary to fit all the data":

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)

# We need a label for each image. We'll use 1 for 3s and 0 for 7s:

train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)

# A Dataset in PyTorch is required to return a tuple of (x,y) when indexed. 
# Python provides a zip function which, when combined with list, provides a simple way to get this functionality:

dset = list(zip(train_x,train_y))

# Do this for the validation set as well

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))

# Now we need an (initially random) weight for every pixel 
# (this is the initialize step in our seven-step process):

def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28,1))

# The function weights*pixels won't be flexible enough
# —it is always equal to 0 when the pixels are equal to 0 
# (i.e., its intercept is 0). 
# You might remember from high school math that the formula for a line is y=w*x+b; 
# we still need the b. We'll initialize it to a random number too:

bias = init_params(1)

# Introduce a function to do matrix multiplication on the weights and parameters as well as the bias
# Khan Academy link on matrix multiplication: https://youtu.be/kT4Mp9EdVqs
# This equation, batch@weights + bias, is one of the two fundamental equations of any neural network 

def linear1(xb): return xb@weights + bias

# call the function

preds = linear1(train_x)

# Let's check our accuracy. 
# To decide if an output represents a 3 or a 7, 
# we can just check whether it's greater than 0.0, 
# so our accuracy for each item can be calculated (using broadcasting, so no loops!) 
# with:

corrects = (preds>0.0).float() == train_y
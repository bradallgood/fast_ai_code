#------------------------------------------------------------------------------
# Imports and data collection
#------------------------------------------------------------------------------
import os
from pathlib import Path

path = Path('titanic')
if not path.exists():
   import zipfile, kaggle
   kaggle.api.competition_download_cli(str(path))
   zipfile.ZipFile(f'{path}.zip').extractall(path)

import torch, numpy as np, pandas as pd
from torch import tensor
from fastai.data.transforms import RandomSplitter
import sympy
#------------------------------------------------------------------------------
#Functions
#------------------------------------------------------------------------------
#Calculate preditions - Updated to use the sigmoid function to get vals between
#0 and 1
#def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))

#Calculate loss
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()

#Updae the coeffs
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()

#Run one session 
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")

#Initialize the coeffs
def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()

def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs

#Quick way to look at the coefficient values for each independent variable
def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))

# Calculate accuracy
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
#------------------------------------------------------------------------------
#Main
#------------------------------------------------------------------------------
# Read in the trainng dataset
df = pd.read_csv(path/'train.csv')

#Replace blank values with the mode for that column
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)

#Scale the fare column logrimticly
df['LogFare'] = np.log(df['Fare']+1)

#Set non numeric catagories to one hot
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])

#Set your dependent variable or lable
t_dep = tensor(df.Survived)

#Set your independent variables
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
t_indep = tensor(df[indep_cols].values, dtype=torch.float)

#Set up your coeffiencts
torch.manual_seed(442)
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5

#Since the age column is much larger numbers than the rest lets divide every 
#column by its max value
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals

#Calculate inital predictions
#Note: no bias is needed since we have columns that have 1 in at least on of
#them
preds = (t_indep*coeffs).sum(axis=1)

#lets calculate the loss from the predictions
loss = torch.abs(preds-t_dep).mean()

#NOTE: the last two steps have been defined as functions now
#calc_pres and calc_loss

#Now lets do a gradient descent step, first set up the coeffs to have grad
coeffs.requires_grad_()

loss = calc_loss(coeffs,t_indep,t_dep)

#Now calculate the gradients nows
loss.backward()

#Update the coefficents with the gradient plus learning rate
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()

#Now lets get ready to train and validate the model
#First lets split our data into training and validation steps
trn_split,val_split=RandomSplitter(seed=42)(df)

#Then lets sort the data (dependent and independent)
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]

#Now lets use our new functions (train_model, init_coeffs, one_epoch, update_coeffs)
# to run a real model
coeffs = train_model(18, lr=0.2)




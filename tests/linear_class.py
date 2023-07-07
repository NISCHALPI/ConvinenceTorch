#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05)


# In[3]:


sns.scatterplot(x=X[:,0] , y=X[:,1], hue=y)


# In[4]:


from torch.utils.data import TensorDataset, DataLoader, random_split
X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.int64)


# 

# In[5]:


dataset = TensorDataset(X, y)

train , val = random_split(dataset, lengths=[0.8,0.2])
trainloader, valloader = DataLoader(train) , DataLoader(val) 


# In[6]:


in_features = 2
out_features = 2
model = torch.nn.Sequential(torch.nn.Linear(2,30), torch.nn.Tanh(), torch.nn.Linear(30,30) , torch.nn.Tanh(), torch.nn.Linear(30,2))
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# In[7]:


from src.torchutils.skeletons import Register
from src.torchutils.trainer import NNtrainer


# In[8]:


trainer  = NNtrainer(model=model, optimizer=optimizer, loss=loss)


# In[9]:


trainer.train(trainloader=trainloader, valloader=valloader, epoch=20,  metrics=['accuracy'], record_loss=True, checkpoint_file='train'  , checkpoint_every_x=2)
trainer.plot_train_validation_metric_curve('accuracy')


# In[10]:


sns.scatterplot(x=X[:,0] , y=X[:,1], hue=y)
sns.scatterplot(x=X[:,0] , y=X[:,1], hue=trainer.predict(X).argmax(dim=1))


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import os
from torch.utils.data import DataLoader,TensorDataset
from torchvision.transforms import ToTensor
from models.FeatureSetEmbedding import FeatureSetEmbedder

max_feature = 4
embed_dim = 3

data = pd.read_csv("data/airlines.csv")
data = data.drop(['Flight', 'Time','Length'], axis=1)

le = LabelEncoder()
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col]=le.fit_transform(data[col])

X = data.loc[:, ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']]
y = data.Delay

output_dim = y.nunique()

numberOfInstance = X.shape[0]

for index, record in X.iterrows():
    inp = torch.tensor([(0,record['Airline']), (1,record['AirportFrom']), (2, record['AirportTo']), (3, record['DayOfWeek'])])
    if index==0: 
        X_tensor= inp
    else:
        X_tensor= torch.cat((X_tensor,inp),0)

X_tensor=X_tensor.reshape(numberOfInstance,max_feature,2)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

y_tensor = torch.tensor(data['Delay'].values, dtype=torch.long, device=device)

X_train, X_ee, y_train, y_ee = train_test_split(X_tensor, y_tensor, test_size=0.25, random_state=44)

train_dataset= TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_ee,y_ee)

train_dataloader = DataLoader(train_dataset,batch_size= 10)
test_dataloader = DataLoader(test_dataset,batch_size= 10)


model = FeatureSetEmbedder(max_n_features= max_feature,embedding_dim= embed_dim, n_output= output_dim) 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


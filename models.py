import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class sudokuCnn(nn.Module):
  def __init__(self,output_classes,in_channels = 1):
    super(sudokuCnn,self).__init__()
    self.conv1 = nn.Conv2d(1,32,kernel_size = 5,stride = 1,padding = 1)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32,32,kernel_size = 3, stride = 1,padding = 1)
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(1152,128)
    self.dropout1 = nn.Dropout(p=0.2)
    self.fc2 = nn.Linear(128,64)
    self.output = nn.Linear(64, output_classes)

  def forward(self,x):
    x=x.float()
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool2(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.output(x)
    x = F.softmax(x)

    return x

def predict(model,x):
    device = torch.device("cuda")
    model.eval()
    model = model.to(device = device)
    with torch.no_grad():
      x.to(device)
      scores = model(x)
      prediction = scores.argmax(1)

    return prediction

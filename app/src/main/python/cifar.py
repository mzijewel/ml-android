import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.mobile_optimizer import optimize_for_mobile

import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import time
import platform,sys,os
from os.path import dirname, join

PY_DIR=dirname(__file__)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def predict(model,img_path):
  with torch.no_grad():
    img=Image.open(img_path).resize((32,32))
    # img_tns=torchvision.transforms.ToTensor()(img).unsqueeze(0)
    img_tns=transform(img).unsqueeze(0)
    result=model(img_tns)
    probabilities=torch.softmax(result[0],dim=0)
    # max_indx=torch.argmax(probabilities)
    max_val,max_indx=torch.max(probabilities,dim=0)
    item=classes[max_indx]
    print(f'{img_path:10s} : {item:5s} : {max_val*100:0.2f}% : index {max_indx}')


def load_weight(path):
  model = Model()
  model.load_state_dict(torch.load(path))
  return model


def load_model(path):
  model=torch.jit.load(path)
  model.eval()
  # count=sum(p.numel() for p in model.parameters())
  # print(f"prams: {count}")
  return model


def test_model():
    st=time.time()
    model_path=f'{PY_DIR}/cifar73.pt'
    model=load_model(model_path)
    imgs=["car.jpeg","cat.jpg","deer.jpeg","dog.jpg"]
    for img in imgs:
        img_path=f'{PY_DIR}/data/img/{img}'
        predict(model,img_path)
    et=time.time()
    print(f'Total time:{et-st:.3f}')

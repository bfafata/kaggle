#clean data
#model
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from preprocess import preprocess
from model import model1
from train import train_model

from torch.utils.tensorboard import SummaryWriter





train_path = "train.csv"
test_path = "test.csv"

train_frame = pd.read_csv(train_path)
test_frame = pd.read_csv(train_path)

def main():
    train_inputs,train_labels = preprocess(train_frame)
    test_inputs,test_labels = preprocess(test_frame)

    #params
    input_size = 35
    hidden_size = 64
    output_size = 2
    
    model = model1(input_size,hidden_size,output_size).double()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.2)
    #
    writer = SummaryWriter('runs/experiment_4')
    #
    num_epochs = 200
    train_model(model,criterion,optimizer,train_inputs,train_labels,writer,num_epochs)

    


if __name__ == "__main__":
    main()

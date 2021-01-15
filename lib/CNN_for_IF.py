import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN2D_micromotion(nn.Module):
    # W_out = (W_input - filter_size + 2*padding) / stride + 1
    def __init__(self, no_classes = 2, dropout=0.5):
        super(CNN2D_micromotion, self).__init__() # super -> odnosi się do klasy matki, czyli nn.Module
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0),
            # W_out = (151 - 5 + 2*0) / 2 + 1 = 74
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=4, stride=1, padding=0),
            # W_out = (74 - 5 + 2*0) / 1 + 1 = 70
            nn.AvgPool2d(kernel_size=5, stride=5, padding=0),
            # W_out = (70 - 5 + 2*0) / 5 + 1 = 14
            nn.ReLU())
        #self.drop_out = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(14 * 14 * 8, no_classes) # output: no_classes
        #softmax

    def forward(self, x):
        #print("Begin at: ", x.shape)
        x = self.layer1(x)
        #print("1st layer, 5x228: ", x.shape)
        x = self.layer2(x)
        #print("2nd layer, 8x56: ", x.shape)
        x = x.reshape(-1, 14 * 14 * 8)
        #x = self.drop_out(x)
        x = self.fc1(x)
        return x

class CNN2D(nn.Module):
    # W_out = (W_input - filter_size + 2*padding) / stride + 1
    def __init__(self, no_classes = 2, dropout = 0.5):
        super(CNN2D, self).__init__() # super -> odnosi się do klasy matki, czyli nn.Module
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=0),
            # W_out = (56 - 5 + 2*0) / 1 + 1 = 52
            nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
            # W_out = (52 - 4 + 2*0) / 2 + 1 = 25
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=0),
            # W_out = (25 - 5 + 2*0) / 1 + 1 = 21
            nn.AvgPool2d(kernel_size=5, stride=4, padding=0),
            # W_out = (21 - 5 + 2*0) / 4 + 1 = 5
            nn.ReLU())
        #self.drop_out = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(5 * 5 * 8, no_classes) # output: no_classes
        #softmax

    def forward(self, x):
        #print("Begin at: ", x.shape)
        x = self.layer1(x)
        #print("1st layer, 5x228: ", x.shape)
        x = self.layer2(x)
        #print("2nd layer, 8x56: ", x.shape)
        x = x.reshape(-1, 5 * 5 * 8)
        #x = self.drop_out(x)
        x = self.fc1(x)
        return x
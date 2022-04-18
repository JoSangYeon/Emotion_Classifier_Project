# https://github.com/HideOnHouse/TorchBase

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import MyDataset
from model import get_Model
from learning import train, evaluate, calc_acc
from inference import inference

def label2int(data, label_tags):
    for i in range(len(data)):
        data.iloc[i,1] = label_tags.index(data.iloc[i, 1])
    return data


def draw_history(history):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2,1,1)
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    plt.show()

def main():
    # label_tags
    label_tags = ['불안', '슬픔', '상처', '당황', '분노', '기쁨']

    train_path = "train.csv"
    test_path = "test.csv"

    train_data = pd.read_csv(train_path, encoding='cp949')
    test_data = pd.read_csv(test_path, encoding='cp949')

    train_data = label2int(train_data, label_tags)
    test_data = label2int(test_data, label_tags)

    # your Data Pre-Processing
    train_x, train_y = train_data.iloc[:, :1], train_data.iloc[:, 1:]
    test_x, test_y = test_data.iloc[:, :1], test_data.iloc[:, 1:]

    # data split
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, random_state=17, test_size=0.05)

    # Check Train, Valid, Test Image's Shape
    print("The Shape of Train Input: ", train_x.shape)
    print("The Shape of Valid Input: ", valid_x.shape)
    print("The Shape of Test Input: ", test_x.shape)

    # Check Train, Valid Label's Shape
    print("The Shape of Train Labels: ", train_y.shape)
    print("The Shape of Valid Labels: ", valid_y.shape)
    print("The Shape of Valid Labels: ", test_y.shape)

    # Create Dataset and DataLoader
    train_dataset = MyDataset(train_x, train_y)
    valid_dataset = MyDataset(valid_x, valid_y)
    test_dataset = MyDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model_name = "MyModel_1"
    model = get_Model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # train
    print("============================= Train =============================")
    history = train(model, device, optimizer, criterion, 10, train_loader, valid_loader)

    # Test
    print("============================= Test =============================")
    test_loss, test_acc = evaluate(model, device, criterion, test_loader)
    print("test loss : {:.6f}".format(test_loss))
    print("test acc : {:.3f}".format(test_acc))

    file_name = model_name
    torch.save(model, f"models/{file_name}.pt")
    with open(f"models/{file_name}_history.pickle", 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    # print(history)
    draw_history(history)

    # # Inference
    # # print("=========================== Inference ===========================")
    # # inference(device, criterion, infer_dataloader)


if __name__ == '__main__':
    main()
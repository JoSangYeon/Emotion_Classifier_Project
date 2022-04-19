import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast, AutoModel, AutoTokenizer

def label2int(data, label_tags):
    for i in range(len(data)):
        data.iloc[i,1] = label_tags.index(data.iloc[i, 1])
    return data

"""
class MyDataset(Dataset):
    def __init__(self, x_data, y_data, max_len=128):
        super(MyDataset, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.data = pd.concat([x_data, y_data], axis=1)
        self.max_len = max_len
        # self.tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=True)
        self.inputs = [self.convert_token([self.data.iloc[i]['sentence']]) for i in range(len(self.data))]
        self.label = [np.int32(self.data.iloc[i]['label']) for i in range(len(self.data))]

    def convert_token(self, text):
        token = self.tokenizer.encode(text[0])
        attention_mask = [1] * len(token) + [0] * (self.max_len - len(token))
        token = token + self.tokenizer.convert_tokens_to_ids(["[PAD]"] * (self.max_len - len(token)))
        return [np.int32(token), np.int32(attention_mask)]

    def __getitem__(self, idx):
        return self.inputs[idx][0], self.inputs[idx][1], self.label[idx]

    def __len__(self):
        return len(self.label)
"""

class MyDataset_1(Dataset):
    def __init__(self, x_data, y_data, max_length=128, padding='max_length', num_classes=6):
        super(MyDataset_1, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes

        self.tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = True
        self.return_attention_mask = True

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        ## sentence Tokenize ##
        x = self.sentence.iloc[idx][0]
        tokenizer_output = self.tokenizer(x, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        token_type_ids = tokenizer_output.token_type_ids

        ## sentiment ##
        y = torch.tensor(self.sentiment.iloc[idx][0])
        y = F.one_hot(y, num_classes = self.num_classes).float()
        # y = np.int32(self.sentiment.iloc[idx][0])

        return input_ids[0], attention_mask[0], token_type_ids[0], y

    def show_item(self, idx=0):
        feature1, feature2, label = self.__getitem__(idx)

        print("input_ids's Shape : {}".format(feature1.shape))
        print("attention_mask's Shape : {}".format(feature2.shape))
        # print("token_type_ids's Shape : {}".format(feature[2].shape))
        print("Label's Shape : {}".format(label.shape))

        return feature1, feature2, label
    
class MyDataset_2(Dataset):
    def __init__(self, x_data, y_data, max_length=128, padding='max_length', num_classes=6):
        super(MyDataset_2, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes

        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = True
        self.return_attention_mask = True

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        ## sentence Tokenize ##
        x = self.sentence.iloc[idx][0]
        tokenizer_output = self.tokenizer(x, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        token_type_ids = tokenizer_output.token_type_ids

        ## sentiment ##
        y = torch.tensor(self.sentiment.iloc[idx][0])
        y = F.one_hot(y, num_classes = self.num_classes).float()
        # y = np.int32(self.sentiment.iloc[idx][0])

        return input_ids[0], attention_mask[0], token_type_ids[0], y

    def show_item(self, idx=0):
        feature1, feature2, label = self.__getitem__(idx)

        print("input_ids's Shape : {}".format(feature1.shape))
        print("attention_mask's Shape : {}".format(feature2.shape))
        # print("token_type_ids's Shape : {}".format(feature[2].shape))
        print("Label's Shape : {}".format(label.shape))

        return feature1, feature2, label


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

    train = MyDataset_2(train_x, train_y)

    a, b, c, d = train.__getitem__(10)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d)

    print(type(a))
    print(type(b))
    print(type(c))
    print(type(d))

if __name__ == "__main__":
    main()
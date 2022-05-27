import numpy as np
import pandas as pd
import random
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

class MyDataset(Dataset):
    def __init__(self, 
                 x_data, 
                 y_data,
                 model_path="klue/bert-base",
                 max_length=128, 
                 padding='max_length',
                 num_classes=3):
        super(MyDataset, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = False
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
        # token_type_ids = tokenizer_output.token_type_ids

        ## sentiment ##
        y = torch.tensor(self.sentiment.iloc[idx][0])
        y = F.one_hot(y, num_classes = self.num_classes).float()
        # y = np.int32(self.sentiment.iloc[idx][0])

        return input_ids[0], attention_mask[0], y

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("input_ids's Shape : {}".format(feature1.shape))
        print("attention_mask's Shape : {}".format(feature2.shape))
        print("token_type_ids's Shape : {}".format(feature[2].shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label

    
class MyDataset_Contrastive(Dataset):
    def __init__(self, 
                 x_data, 
                 y_data,
                 model_path="klue/bert-base",
                 max_length=128, 
                 padding='max_length',
                 num_classes=3):
        super(MyDataset_Contrastive, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes
        self.index = np.array(range(len(self.sentence))) # [0, 1, ,,, , n-1, n]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = False
        self.return_attention_mask = True
        

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        ## sentence Tokenize ##
        ### Target Sentence ###
        x = self.sentence.iloc[idx][0]
        tokenizer_output = self.tokenizer(x, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        x_input_ids = tokenizer_output.input_ids
        x_attention_mask = tokenizer_output.attention_mask
        # x_token_type_ids = tokenizer_output.token_type_ids
        
        ## sentiment ##
        ### Target Label ###
        y = torch.tensor(self.sentiment.iloc[idx][0])
        y = F.one_hot(y, num_classes = self.num_classes).float()
        
        
        ## Contrastive Data ##
        contra_idx = np.random.choice(self.index[self.index != idx])
        ### Contrastive Sentence ###
        c = self.sentence.iloc[contra_idx][0]
        tokenizer_output = self.tokenizer(c, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        c_input_ids = tokenizer_output.input_ids
        c_attention_mask = tokenizer_output.attention_mask
        # c_token_type_ids = tokenizer_output.token_type_ids

        ### Contrastive Label ###
        y0 = self.sentiment.iloc[idx][0]
        y1 = self.sentiment.iloc[contra_idx][0]

        c_y = 1 if y0 == y1 else 0 # 같으면 1 다르면 0
        return (x_input_ids[0], x_attention_mask[0]), (c_input_ids[0], c_attention_mask[0]), (y, c_y)
    

class MyDataset_Triplet(Dataset):
    def __init__(self, 
                 x_data, 
                 y_data,
                 model_path="klue/bert-base",
                 max_length=128, 
                 padding='max_length',
                 num_classes=3):
        super(MyDataset_Triplet, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes
        self.index = np.array(range(len(self.sentence))) # [0, 1, ,,, , n-1, n]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = False
        self.return_attention_mask = True
        

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        ## Anchor ##
        ### Sentence ###
        anchor_sentence = self.sentence.iloc[idx][0]
        anchor_token = self.tokenizer(anchor_sentence, max_length=self.max_length,
                                          padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        anchor_input_ids = anchor_token.input_ids[0]
        anchor_attention_mask = anchor_token.attention_mask[0]
        # x_token_type_ids = tokenizer_output.token_type_ids
        ### Label ###
        anchor_label = torch.tensor(self.sentiment.iloc[idx][0])
        anchor_label = F.one_hot(anchor_label, num_classes = self.num_classes).float()
        
        ## Triplet ##
        label = self.sentiment.iloc[idx][0] # 현재 idx의 label 값
        query = self.index!=idx
        ### Positive ###
        pos_query = (self.sentiment==label).to_numpy().reshape(-1)
        pos_index = self.index[query & pos_query]
        pos_idx = np.random.choice(pos_index)
        
        pos_sentence = self.sentence.iloc[pos_idx][0]
        pos_token = self.tokenizer(pos_sentence, max_length=self.max_length,
                                          padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)
        
        pos_input_ids = pos_token.input_ids[0]
        pos_attention_mask = pos_token.attention_mask[0]
        
        ### Negative ###
        neg_query = (self.sentiment!=label).to_numpy().reshape(-1)
        neg_index = self.index[query & neg_query]
        neg_idx = np.random.choice(neg_index)

        neg_sentence = self.sentence.iloc[neg_idx][0]
        neg_token = self.tokenizer(neg_sentence, max_length=self.max_length,
                                          padding=self.padding,
                                          return_tensors=self.return_tensors,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)
        
        neg_input_ids = neg_token.input_ids[0]
        neg_attention_mask = neg_token.attention_mask[0]
        
        return (anchor_input_ids, anchor_attention_mask), (pos_input_ids, pos_attention_mask), (neg_input_ids, neg_attention_mask), anchor_label
    

def main():
    # label_tags
    label_tags = ['불안', '슬픔', '기쁨']

    train_path = "train.csv"
    test_path = "test.csv"

    train_data = pd.read_csv(train_path, encoding='cp949')
    test_data = pd.read_csv(test_path, encoding='cp949')

    train_data = label2int(train_data, label_tags)
    test_data = label2int(test_data, label_tags)

    # your Data Pre-Processing
    train_x, train_y = train_data.iloc[:, :1], train_data.iloc[:, 1:]
    test_x, test_y = test_data.iloc[:, :1], test_data.iloc[:, 1:]

    train = MyDataset(train_x, train_y)

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
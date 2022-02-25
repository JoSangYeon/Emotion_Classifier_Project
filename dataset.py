import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, max_length=128, padding='max_length', num_classes=6):
        super(MyDataset, self).__init__()
        self.sentence = x_data
        self.sentiment = y_data
        self.num_classes = num_classes

        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

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

        return (input_ids[0], attention_mask[0], token_type_ids[0]), y

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("input_ids's Shape : {}".format(feature[0].shape))
        print("attention_mask's Shape : {}".format(feature[1].shape))
        print("token_type_ids's Shape : {}".format(feature[2].shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label


def main():
    pass

if __name__ == "__main__":
    main()
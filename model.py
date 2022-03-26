import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from transformers import BertModel, AutoModel, AutoTokenizer

class MyModel_1(nn.Module):
    def __init__(self, num_classes=6):
        super(MyModel_1, self).__init__()

        self.bert = AutoModel.from_pretrained("kykim/bert-kor-base")
        # self.bert = AutoModel.from_pretrained("klue/bert-base")
        self.bert.requires_grad = True

        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        # total_vector = bert_output.last_hidden_state        # (batch, 128, 768)
        cls_vector = bert_output.pooler_output              # (batch, 768)

        x = self.fc1(cls_vector)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        return x

def get_Model(class_name):
    try:
        Myclass = eval(class_name)()
        return Myclass
    except NameError as e:
        print("Class [{}] is not defined".format(class_name))

def main():
    model = get_Model("MyModel_1").cuda()

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    input_data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    summary(model, input_size=[(128,), (128,), (128,)], device='cuda')


if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

class MyModel_1(nn.Module):
    def __init__(self, 
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_1, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        cls_vector = bert_output.pooler_output              # (batch, 768)

        x_embed = self.fc1(cls_vector)
        x = self.act_fn(x_embed)
        x = self.drop(x)

        x = self.fc2(x)
        return x, x_embed
    

class MyModel_2(nn.Module):
    def __init__(self, 
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_2, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        all_embed = bert_output.last_hidden_state        # All Token Embed : (batch, 128, 768)
        mean_pool = self.mean_pooling(all_embed, attention_mask) # (batch, 768)

        x_embed = self.fc1(mean_pool)
        x = self.act_fn(x_embed)
        x = self.drop(x)

        x = self.fc2(x)
        return x, x_embed
    
class MyModel_3(nn.Module):
    def __init__(self, 
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_3, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)
        
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.contra = nn.Linear(768, 256)
        
        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()
        

    def forward(self, x_inputs, x_atts, c_inputs=None, c_atts=None):
        x_output = self.bert(x_inputs, x_atts)
        x_embed = x_output.pooler_output             # CLS Token Embed : (batch, 768)
        
        x = self.fc1(x_embed)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x_embed = self.contra(x_embed)
        
        if (c_inputs is not None) and (c_atts is not None):
            c_output = self.bert(c_inputs, c_atts)
            c_embed = c_output.pooler_output         # CLS Token Embed : (batch, 768) 
            c_embed = self.contra(c_embed)
            return x, x_embed, c_embed
        else:
            return x, x_embed
    
    
class MyModel_4(nn.Module):
    def __init__(self, 
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_4, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)
        
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.contra = nn.Linear(768, 256)
        
        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x_inputs, x_atts, c_inputs=None, c_atts=None):
        x_output = self.bert(x_inputs, x_atts)
        x_embed = x_output.last_hidden_state         # All Token Embed : (batch, 128, 768)
        x_embed = self.mean_pooling(x_embed, x_atts) # (batch, 768)
        
        x = self.fc1(x_embed)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x_embed = self.contra(x_embed)
        
        if (c_inputs is not None) and (c_atts is not None):
            c_output = self.bert(c_inputs, c_atts)
            c_embed = c_output.last_hidden_state         # All Token Embed : (batch, 128, 768)
            c_embed = self.mean_pooling(c_embed, c_atts) # (batch, 768)
            c_embed = self.contra(c_embed)
            return x, x_embed, c_embed
        else:
            return x, x_embed
        

class MyModel_5(nn.Module):
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_5, self).__init__()
        self.device=device

        self.bert = AutoModel.from_pretrained(model_path)
        
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.toEmbed = nn.Linear(768, 256)
        
        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()

    def forward(self, x_inputs, x_atts, p_inputs=None, p_atts=None, n_inputs=None, n_atts=None):
        x_output = self.bert(x_inputs, x_atts)
        x_embed = x_output.pooler_output             # CLS Token Embed : (batch, 768)            
        
        x = self.fc1(x_embed)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x_embed = self.toEmbed(x_embed)          # (batch, 768) -> (batch, 256)
        
        if (p_inputs is not None) and (n_inputs is not None):
            p_output = self.bert(p_inputs, p_atts)
            p_embed = p_output.pooler_output         # CLS Token Embed : (batch, 768) 
            
            n_output = self.bert(n_inputs, n_atts)
            n_embed = n_output.pooler_output         # CLS Token Embed : (batch, 768) 
            
            p_embed = self.toEmbed(p_embed)          # (batch, 768) -> (batch, 256)
            n_embed = self.toEmbed(n_embed)          # (batch, 768) -> (batch, 256)
            return x, x_embed, p_embed, n_embed
        else:
            return x, x_embed
    
    
class MyModel_6(nn.Module):
    def __init__(self, 
                 model_path="klue/bert-base",
                 num_classes=3):
        super(MyModel_6, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)
        
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.toEmbed = nn.Linear(768, 256)
        
        self.drop = nn.Dropout(p=0.2)
        self.act_fn = nn.ReLU()
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x_inputs, x_atts, p_inputs=None, p_atts=None, n_inputs=None, n_atts=None):
        x_output = self.bert(x_inputs, x_atts)
        
        x_embed = x_output.last_hidden_state         # All Token Embed : (batch, 128, 768)
        x_embed = self.mean_pooling(x_embed, x_atts) # (batch, 768)
        
        x = self.fc1(x_embed)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x_embed = self.toEmbed(x_embed)          # (batch, 768) -> (batch, 256)
        
        if (p_inputs is not None) and (n_inputs is not None):
            p_output = self.bert(p_inputs, p_atts)
            p_embed = p_output.last_hidden_state     # All Token Embed : (batch, 128, 768)
            p_embed = self.mean_pooling(p_embed, p_atts) # mean pooling : (batch, 768)
            
            n_output = self.bert(n_inputs, n_atts)
            n_embed = n_output.last_hidden_state     # All Token Embed : (batch, 128, 768)
            n_embed = self.mean_pooling(n_embed, n_atts) # mean pooling : (batch, 768)
            
            p_embed = self.toEmbed(p_embed)          # (batch, 768) -> (batch, 256)
            n_embed = self.toEmbed(n_embed)          # (batch, 768) -> (batch, 256)
            return x, x_embed, p_embed, n_embed
        else:
            return x, x_embed


def get_Model(class_name, model_path):
    try:
        Myclass = eval(class_name)(model_path=model_path)
        return Myclass
    except NameError as e:
        print("Class [{}] is not defined".format(class_name))

def main():
    model = get_Model("MyModel_1").cuda()


if __name__ == "__main__":
    main()
import os
import torch
from tqdm import tqdm
from learning import evaluate

def inference(model, tokenizer, inputs):
    label_tags['불안', '슬픔', '기쁨']
    tokenize_output = tokenizer(inputs, max_length=128, padding='max_length', 
                         return_tensors='pt', return_token_type_ids=False, 
                         return_attention_mask=True)
    predict = model(**tokenize_output)
    p_v, p_i = torch.max(predict, dim=-1)
    return label_tags[p_i.item()]

def display_result(device, criterion, data_loader):
    file_list = os.listdir(f"models/")

    for file in file_list:
        if file[-2:] != "pt":
            continue

        model = torch.load(f"models/"+file)
        model.to(device); model.eval()

        print("Inference {}".format(file))
        loss, acc = evaluate(model, device, criterion, data_loader)
        print("\tloss : {:.6f}".format(loss))
        print("\tacc : {:.3f}".format(acc))
        print("\n")



def main():
    pass


if __name__ == '__main__':
    main()
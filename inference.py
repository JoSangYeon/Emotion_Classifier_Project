import os
import sys
import torch
from tqdm import tqdm
from learning import evaluate, calc_acc

def inference(model, tokenizer, inputs):
    label_tags['불안', '슬픔', '기쁨']
    tokenize_output = tokenizer(inputs, max_length=128, padding='max_length', 
                         return_tensors='pt', return_token_type_ids=False, 
                         return_attention_mask=True)
    predict = model(**tokenize_output)
    p_v, p_i = torch.max(predict, dim=-1)
    return label_tags[p_i.item()]


def model_eval(model, device, criterion, loader):
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param criterion: loss function
    :param data_loader: valid or test Datasets
    :return: loss and acc
    """
    model.eval()
    total_loss = total_acc = 0

    with torch.no_grad():
        # in notebook
        # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(enumerate(loader), file=sys.stdout)

        for batch_idx, (input_ids, att_mask, target) in pbar:
            input_ids, att_mask = input_ids.to(device), att_mask.to(device)
            target = target.to(device)

            output, _ = model(input_ids, att_mask)
            loss = criterion(output, target)
            acc = calc_acc(output, target)

            total_loss += loss.item()
            total_acc += acc

            acc = total_acc / (batch_idx * loader.batch_size + len(target))
            loss = total_loss / (batch_idx * loader.batch_size + len(target))
            pbar.set_postfix(loss='{:.6f}, acc={:.3f}'.format(loss, acc))
        pbar.close()

    total_loss = total_loss / len(loader)
    total_acc = total_acc / len(loader.dataset)

    return total_loss, total_acc

def display_result(device, criterion, bert_loader, robert_loader):
    file_list = os.listdir(f"models/")
    file_list.sort()

    for file in file_list:
        if file[-2:] != "pt":
            continue
        
        loader = bert_loader if 'BERT' in file else robert_loader
            
        model = torch.load(f"models/"+file)
        model.to(device); model.eval()

        print("Inference {}".format(file))
        loss, acc = model_eval(model, device, criterion, loader)
        print("\tloss : {:.6f}".format(loss))
        print("\tacc : {:.3f}".format(acc))
        print("\n")



def main():
    pass


if __name__ == '__main__':
    main()
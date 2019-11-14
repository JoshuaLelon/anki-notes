import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.BERTSentiment import BERTSentimentClassifier
from datasets.dataset_creator import get_val_dataloader_for_dataset, get_train_dataloader_for_dataset

def train_bert(dataset, freeze_base, max_len, batch_size, learning_rate, print_per_n_lines, max_epochs):
    gpu = 0
    if torch.cuda.is_available():
        gpu = 1
    return "implement me"
    
def evaluate_bert(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        net.cuda(gpu)
        
    start_time = time.time()
    print("Evaluating BERT on the " + dataset + " dataset using the " + str(device) + ".")
    dataloader = get_val_dataloader_for_dataset(dataset)
    criterion = nn.BCEWithLogitsLoss()
    net = BERTSentimentClassifier() # for now, we will automatically do BERT, later will generalize to any 
    net.eval()
    
    mean_accuracy, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attention_masks, labels in dataloader:
            if torch.cuda.is_available():
                seq, attention_masks, labels = seq.cuda(), attention_masks.cuda(), labels.cuda()
            logits = net(seq, attention_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_accuracy += get_accuracy_from_logits(logits, labels)
            count += 1
    
    print("Accuracy: ")
    print(mean_accuracy / count)
    print("Avg loss: ")
    print(mean_loss / count)
    print("Done in {} seconds".format(time.time() - start_time))

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def train_nb(dataset):
    gpu = 0
    if torch.cuda.is_available():
        gpu = 1
    return "implement me"
    
def evaluate_nb(dataset):
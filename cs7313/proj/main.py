import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.BERTSentiment import BERTSentimentClassifier
from datasets.dataset_creator import get_val_dataloader_for_dataset, get_train_dataloader_for_dataset

def train(model, dataset, freeze_base, max_len, batch_size, learning_rate, print_per_n_lines, max_epochs):
    gpu = 0
    if torch.cuda.is_available():
        gpu = 1
    return "implement me"
    
def evaluate(model, dataset):
    gpu = 0
    if torch.cuda.is_available():
        gpu = 1
        
    start_time = time.time()
    print("Evaluating " + model + " on the " + dataset + " dataset.")
    dataloader = get_val_dataloader_for_dataset(dataset)
    criterion = nn.BCEWithLogitsLoss()
    net = BERTSentimentClassifier() # for now, we will automatically do BERT, later will generalize to any model
    net.cuda(gpu)
    net.eval()
    
    mean_accuracy, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attention_masks, labels in dataloader:
            seq, attention_masks, labels = seq.cuda(gpu), attention_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attention_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_accuracy += get_accuracy_from_logits(logits, labels)
            count += 1
    
    print("Accuracy: " + (mean_accuracy / count))
    print("Avg loss: " + (mean_loss / count))
    print("Done in {} seconds".format(time.time() - start_time))

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SentimentClassifier

def train(model, dataset, gpu, freeze_base, max_len, batch_size, learning_rate, print_per_n_lines, max_epochs):

    
def evaluate(model, dataset, gpu):
    start_time = time.time()
    print("Evaluating " + model + " on the " + dataset + " dataset.")
    dataloader = get_val_dataloader_for_dataset(dataset)
    criterion = nn.BCEWithLogitsLoss()
    net = BERTSentiment() # for now, we will automatically do BERT, later will generalize to any model
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

def get_train_val_dataloaders_for_dataset(dataset):
    return get_train_dataloader_for_dataset(dataset), get_val_dataloader_for_dataset(dataset)

def get_train_dataloader_for_dataset(dataset):
    return get_dataloader_from_dataset(dataset, max_length, batch_size, num_workers, "train")

def get_val_dataloader_for_dataset(dataset):
    return get_dataloader_from_dataset(dataset, max_length, batch_size, num_workers, "val")

def get_dataloader_from_dataset(dataset, max_length, batch_size, num_workers, train_or_val):
    filename = DATASET_DIR + train_or_val + "_" + dataset + ".csv"
    pytorch_dataset = get_pytorch_dataset(filename, dataset, max_length)
    dataset_loader = DataLoader(pytorch_dataset, batch_size = batch_size, num_workers = num_workers)
    return dataset_loader

def get_pytorch_dataset(filename, dataset, max_length):
    # here, in the future, we will use the dataset variable to retrieve the custom dataset class for the data
    # for now, we will just use it for one case: twitter (to be built)
    pytorch_dataset = TwitterDataset(filename = filename, maxlen = max_length)
    return pytorch_dataset

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc
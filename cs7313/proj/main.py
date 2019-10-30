import torch
import torch.nn as nn
from model import SentimentClassifier

def train(model, dataset, gpu, freeze_base, max_len, batch_size, learning_rate, print_per_n_lines, max_epochs):

    
def evaluate(model, dataset, gpu):
    print("Evaluating " + model + " on the " + dataset + " dataset.")
    dataloader = get_dataloader_for_dataset(dataset)
    criterion = nn.BCEWithLogitsLoss()
    net = SentimentClassifier()
    net.cuda(gpu)
    net.eval()

def get_dataloader_for_dataset(dataset):
    return dataset
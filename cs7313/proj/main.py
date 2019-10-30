import torch
import torch.nn as nn
from model import SentimentClassifier

def train(model, dataset, gpu, freeze_base, max_len, batch_size, learning_rate, print_per_n_lines, max_epochs):

    
def evaluate(model, dataset, gpu):
    print("Evaluating " + model + " on the " + dataset + " dataset.")
    dataloader = get_dataloader_for_dataset(dataset)
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

def get_dataloader_for_dataset(dataset):
    return dataset

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc
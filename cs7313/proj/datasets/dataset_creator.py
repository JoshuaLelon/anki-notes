import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

DATASET_DIR = "" # TODO

class BERT_Dataset(Dataset):
    
    # This dataloader has already been made for BERT tokenizing, so I'm going to adapt a lot of the code from here:
    # https://github.com/kabirahuja2431/FineTuneBERT/blob/master/src/dataloader.py

    def __init__(self, filename, max_length):
        self.df = pd.read_csv(filename, delimiter = '\t')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.max_length:
            tokens = tokens + ['[PAD]' for _ in range(self.max_length - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.max_length-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label
    
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
    pytorch_dataset = BERT_Dataset(filename = filename, maxlen = max_length)
    return pytorch_dataset
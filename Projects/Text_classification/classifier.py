import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from clean_text import clean_text

print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Model is running on {device}')

dataset_path = 'Projects\\Text_classification\\dataset\\Twitter_Data.csv'

df = pd.read_csv(dataset_path)
df.dropna(axis=0,inplace=True)

df['clean_text'] = df['clean_text'].apply(clean_text)

text = df['clean_text'].tolist()
labels = df['category'].tolist()

class Twitter_Data(Dataset):
    def __init__(self,text,labels):
        self.text = text
        self.labels = labels
    
    def __len__(self):
        return len(labels)
    
    def __getitem__(self,idx):
        return self.text[idx],self.labels[idx]
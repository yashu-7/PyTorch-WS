import torch
import pandas as pd
import torch.utils
from clean_text import clean_text
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

dataset_path = r'Projects\Text_classification\dataset\Twitter_Data.csv'
df = pd.read_csv(dataset_path)

df.dropna(axis=0, inplace=True)

df['clean_text'] = df['clean_text'].apply(clean_text)

text = df['clean_text'].tolist()
labels = df['category'].tolist()

print(len(text), len(labels))

class TwitterData(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.labels[index]

dataset = TwitterData(text, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train, batch_size=16, shuffle=True)
test_loader = DataLoader(test, batch_size=16)

print(f'Length of train: {len(train)}\nLength of test: {len(test)}')
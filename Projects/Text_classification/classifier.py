import torch
import torch.utils
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from clean_text import clean_text
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

dataset_path = r'Projects\Text_classification\dataset\Twitter_Data.csv'
df = pd.read_csv(dataset_path)

df.dropna(axis=0, inplace=True)

df['clean_text'] = df['clean_text'].apply(clean_text)

text = df['clean_text'].tolist()
labels = df['category'].tolist()

# Mapping labels to a valid range: -1 -> 0, 0 -> 1, 1 -> 2
label_mapping = {-1: 0, 0: 1, 1: 2}
labels = [label_mapping[label] for label in labels]

print(len(text), len(labels))

class TwitterData(Dataset):
    def __init__(self, text, labels, vocab, tokenizer):
        self.text = text
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        label = self.labels[index]
        tokenized_text = self.tokenizer(text) 
        numerical_text = [self.vocab[token] for token in tokenized_text] 
        return torch.tensor(numerical_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Tokenizer and Vocabulary
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(text), specials=["<pad>", "<OOV>"])
vocab.set_default_index(vocab["<OOV>"])

# Creating Dataset and DataLoader
dataset = TwitterData(text, labels, vocab, tokenizer)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

def collate_batch(batch):
    text_list, label_list = [], []
    for (_text, _label) in batch:
        text_list.append(_text)
        label_list.append(_label)
    
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.long)
    
    return text_list.to(device), label_list.to(device)

train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test, batch_size=16, collate_fn=collate_batch)

print(f'Length of train: {len(train)}\nLength of test: {len(test)}')

# Define Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab["<pad>"])
        self.fc = nn.Linear(embed_size, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.mean(dim=1)  
        return self.fc(embedded)

vocab_size = len(vocab)
embed_size = 128
num_class = 3 
num_epochs = 10

model = TextClassifier(vocab_size, embed_size, num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss, total_correct = 0, 0
    for text, labels in train_loader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_correct/len(train):.4f}")

# Evaluation
model.eval()
total_correct = 0

with torch.no_grad():
    for text, labels in test_loader:
        output = model(text)
        total_correct += (output.argmax(1) == labels).sum().item()

print(f"Test Accuracy: {total_correct/len(test):.4f}")

# Save the entire model
torch.save(model, r'Projects\models\text_classifier.pth')

# Optionally, save the vocabulary
import pickle
with open(r'Projects\models\vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
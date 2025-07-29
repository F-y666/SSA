import os
import glob
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from tqdm import tqdm



class Vocabulary:


    def __init__(self, freq_threshold=3):

        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        word_counts = Counter()
        for sentence, _ in sentence_list:
            for word in sentence.split(' '):
                word_counts[word] += 1

        idx = 2
        for word, count in word_counts.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = text.split(' ')
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class TextDataset(Dataset):

    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        numericalized_text = self.vocab.numericalize(text)
        return torch.tensor(numericalized_text), torch.tensor(label)


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))

    return torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0), \
        torch.tensor(label_list, dtype=torch.float32), \
        torch.tensor(lengths)



class TextRNN(nn.Module):


    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden).squeeze(1)


class DPCNN(nn.Module):


    def __init__(self, vocab_size, embedding_dim, num_filters, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_region = nn.Conv2d(1, num_filters, (3, embedding_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, text, text_lengths=None):
        x = self.embedding(text)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x);
        x = self.relu(x);
        x = self.conv(x)
        x = self.padding1(x);
        x = self.relu(x);
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze(-1).squeeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)

        return x.squeeze(1)

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px);
        x = self.relu(x);
        x = self.conv(x)
        x = self.padding1(x);
        x = self.relu(x);
        x = self.conv(x)
        x = x + px
        return x



def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    model.train()
    for texts, labels, lengths in iterator:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)

        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for texts, labels, lengths in iterator:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            binary_preds = torch.round(torch.sigmoid(predictions))
            all_preds.extend(binary_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            epoch_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    return epoch_loss / len(iterator), accuracy, f1


def extract_labeled_texts(file_glob_pattern, label):
    texts = []
    files = glob.glob(file_glob_pattern)
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                if isinstance(data, list):
                    for text in data:
                        texts.append((text.strip().replace('\n', ' '), label))
        except json.JSONDecodeError:
    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['textrnn', 'dpcnn'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32,)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = './Inputs&Outputs/enronqa-SSA/Q-R-T-'
    test_stego_pattern = os.path.join(test_dir, 'output_embed_top500.json')
    test_normal_pattern = os.path.join(test_dir, 'outputs-Llama-3.1-8B-Instruct-*.json')

    test_stego_data = extract_labeled_texts(test_stego_pattern, 1)
    test_normal_data = extract_labeled_texts(test_normal_pattern, 0)
    test_data = test_stego_data + test_normal_data

    test_texts_set = {text for text, label in test_data}

    all_data_root_pattern = './Inputs&Outputs/ablation-enronqa-SSA-*/Q-R-T-'
    all_stego_pattern = os.path.join(all_data_root_pattern, 'output_embed_top100.json')
    all_normal_pattern = os.path.join(all_data_root_pattern, 'outputs-Llama-3.1-8B-Instruct-*.json')

    all_stego_data = extract_labeled_texts(all_stego_pattern, 1)
    all_normal_data = extract_labeled_texts(all_normal_pattern, 0)

    train_stego_data = [item for item in all_stego_data if item[0] not in test_texts_set]
    train_normal_data = [item for item in all_normal_data if item[0] not in test_texts_set]



    min_count = int(min(len(train_normal_data), len(train_stego_data)) * 0.03)
    if min_count == 0:
        raise ValueError("None")

    sampled_normal = random.sample(train_normal_data, min_count)
    sampled_stego = random.sample(train_stego_data, min_count)

    train_data = sampled_normal + sampled_stego
    random.shuffle(train_data)

    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(train_data)

    train_dataset = TextDataset(train_data, vocab)
    test_dataset = TextDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1

    if args.model == 'textrnn':
        model = TextRNN(INPUT_DIM, EMBEDDING_DIM, 256, OUTPUT_DIM, 2, True, 0.5).to(device)
    elif args.model == 'dpcnn':
        model = DPCNN(INPUT_DIM, EMBEDDING_DIM, 250, OUTPUT_DIM).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_test_acc = 0
    best_test_f1 = -1
    for epoch in tqdm(range(args.epochs)):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), f'{args.model}-model.pt')
            else:
                torch.save(model.state_dict(), f'{args.model}-model.pt')
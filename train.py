import torch
import RumbleCommentScraper as RCS
import YoutubeCommentScraper as YCS
from .data_loader import EADataLoader
import logging

scraper = RCS.RumbleCommentScraper(username='CS6474', password='georgiatech')
comments = scraper.scrape_comments_from_url('https://rumble.com/v2gcxvy--live-daily-show-louder-with-crowder.html', 50)

for comment in comments:
    print(comment)

## Models
import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np

dataloader = EADataLoader()
dataloader.loadData(
    "data/SampleAmazonDataset.csv", "reviewText", "overall"
)
num_classes = dataloader.num_classes

vocab_size = dataloader.tokenize.vocab_size
padding_id = dataloader.tokenize.pad_token_id
embedding_layer = nn.Embedding(vocab_size, 300, padding_idx=padding_id)

num_classes  = 2

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     

    def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

#Define the computation graph; one layer hidden network
class LSTMMod(nn.Module):
    def __init__(self):
        super(LSTMMod, self).__init__()
        self.lstm = nn.LSTM()
        self.out = nn.LazyLinear(num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=0)
        
    def forward(self, x):
        return self.logSoftmax(self.lstm(x))

lstmMod = LSTMMod()
optimizer = optim.Adam(lstmMod.parameters(), lr=0.1)


def train(model: nn.Module, iterator: EADataLoader, optimizer: optim, criterion, cust_metrics: dict, custScores:dict, num_classes: int, epochs: int = 15, check: bool = False):
        bad = 0
        prev_train_acc = 0
        iters = 0
        while bad < 5 and iters < epochs:
            model.train()
            train_epoch_loss = 0
            train_epoch_acc = 0
            train_epoch_f1 = 0
            for idx, (tokens, masks, labels, categories) in enumerate(iterator.train_loader):
                optimizer.zero_grad()

                # Necessary locally not sure why
                tokens = tokens.long()
                tokens = tokens.squeeze(1)
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                masks = masks.squeeze(0)

                # Forward pass and
                output = model(tokens)
                probs = nn.functional.softmax(
                    output, -1
                )  # Softmax over final dimension for classification
                preds = torch.argmax(probs, -1)
                preds = preds.to(self.device)
                labels = labels.long()

                # Loss and Optimizer step
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                optimizer.step()

                acc = accuracy_torch(preds, labels, num_classes, device=self.device)
                f1 = f1_torch(preds, labels, num_classes, device=self.device)
                train_epoch_f1 += f1
                train_epoch_loss += loss.item()
                train_epoch_acc += acc

                if check:
                    return

            # Normalize training loss and accuracies
            train_epoch_acc = train_epoch_acc / len(iterator.train_loader)
            train_epoch_loss = train_epoch_loss / len(iterator.train_loader)
            train_epoch_f1 = train_epoch_f1 / len(iterator.train_loader)
            logging.info(f"Epoch: {str(iters).zfill(2)}, Train Accuracy: {train_epoch_acc:.4f}, Train Loss: {train_epoch_loss:.4f}, Train F1: {train_epoch_f1:.4f}")

            # Evaluate on validation set
            val_epoch_loss, val_epoch_acc, val_epoch_f1 = self.eval(model, iterator, criterion, cust_metrics, custScores, num_classes, val=True, check=check)
            logging.info(f"Epoch: {str(iters).zfill(2)}, Val   Accuracy: {val_epoch_acc:.4f}, Val   Loss: {val_epoch_loss:.4f}, Val   F1: {val_epoch_f1:.4f}")

            if train_epoch_acc < prev_train_acc:
                bad += 1
            prev_train_acc = train_epoch_acc
            iters += 1
        return model, train_epoch_acc, train_epoch_acc

# Train
model, epoch_loss, epoch_acc = train(lstmMod, dataloader, optimizer, nn.CrossEntropyLoss(), {}, {}, num_classes, check=True)


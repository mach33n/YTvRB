import torch
import RumbleCommentScraper as RCS
import YTScraper as YCS
from data_loader import EADataLoader
import logging
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
import pickle

# scraper = RCS.RumbleCommentScraper(username='CS6474', password='georgiatech')
# comments = scraper.scrape_comments_from_url('https://rumble.com/v2gcxvy--live-daily-show-louder-with-crowder.html', 50)

# for comment in comments:
#     print(comment)

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


num_classes  = 2

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.embedding_layer = nn.Embedding(vocab_size, 300, padding_idx=padding_id)
         self.linear = torch.nn.Linear(input_dim, output_dim)     

    def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

#Define the computation graph; one layer hidden network
class LSTMMod(nn.Module):
    def __init__(self):
        super(LSTMMod, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, 300, padding_idx=padding_id)
        self.lstm = nn.LSTM(300, 100, dropout=0.3)
        self.batch_norm = nn.BatchNorm1d(300)
        self.relu = nn.ReLU()
        self.out = nn.LazyLinear(num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=0)
        
    def forward(self, x):
        x, hid = self.lstm(self.embedding_layer(x))
        x = self.batch_norm(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        return self.out(x)

lstmMod = LSTMMod()
optimizer = optim.Adam(lstmMod.parameters(), lr=1e-3, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(dataloader.class_weights).float())

def accuracy_torch(
        preds, labels, num_classes: int, average: str = "micro", device: str = None
) -> float:
    """Accuracy for tensors
    Args:
        preds (torch.Tensor): predictions as a (N,) tensor
        labels (torch.Tensor): labels as a (N,) tensor
    Returns:
        float: accuracy score
    """
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average=average)
    acc_metric = acc_metric.to(device)
    acc_tensor = acc_metric(preds, labels)
    return acc_tensor.item()


def f1_torch(
        preds, labels, num_classes: int, average: str = "weighted", device: str = None
) -> float:
    """F1 Score for tensors
    Args:
        preds (torch.Tensor): predictions as a (N,) tensor
        labels (torch.Tensor): labels as a (N,) tensor
        num_classes (int): number of classes in the classification
        average (str, optional): method for aggregating over labels. Defaults to "macro".
            See for details: https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html?highlight=F1
    Returns:
        float: F1 score
    """
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average)
    f1_metric = f1_metric.to(device)
    f1_tensor = f1_metric(preds, labels)
    return f1_tensor.item()

def evalm(model: nn.Module, iterator: EADataLoader, criterion, cust_metrics: dict, custScores:dict, num_classes: int, val: bool = False, check: bool = False):
    model.eval()

    if val:
        loader = iterator.val_loader
    else:
        loader = iterator.test_loader
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    device = "cpu"

    for idx, (tokens, masks, labels) in enumerate(loader):
        # Necessary locally not sure why
        tokens = tokens.long()
        tokens = tokens.squeeze(1)
        tokens = tokens.to(device)
        labels = labels.to(device)
        
        output = model(tokens)

        # Forward pass and
        probs = nn.functional.softmax(output, -1)  # Softmax over final dimension for classification
        preds = torch.argmax(probs, -1)
        preds = preds.to(device)
        labels = labels.long()

        # Loss and Optimizer step
        loss = criterion(output, labels)

        acc = accuracy_torch(preds, labels, num_classes, device=device)
        f1 = f1_torch(preds, labels, num_classes, device=device)
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1

    # Normalize loss and accuracies
    epoch_acc = epoch_acc / len(loader)
    epoch_loss = epoch_loss / len(loader)
    epoch_f1 = epoch_f1 /  len(loader)
    print(f"Epoch acc {epoch_acc}")
    print(f"Epoch loss {epoch_loss}")

    # custScores["f1_score"] = epoch_f1

    if not val:
        return epoch_loss, epoch_acc, epoch_f1
    return epoch_loss, epoch_acc, epoch_f1

def train(model: nn.Module, iterator: EADataLoader, optimizer: optim, criterion, cust_metrics: dict, custScores:dict, num_classes: int, epochs: int = 15, check: bool = False):
        bad = 0
        prev_train_acc = 0
        iters = 0
        device = "cpu"
        while bad < 5 and iters < epochs:
            model.train()
            train_epoch_loss = 0
            train_epoch_acc = 0
            train_epoch_f1 = 0
            for idx, (tokens, masks, labels) in enumerate(iterator.train_loader):
                optimizer.zero_grad()

                # Necessary locally not sure why
                tokens = tokens.long()
                tokens = tokens.squeeze(1)
                tokens = tokens.to(device)
                labels = labels.to(device)
                masks = masks.squeeze(0)

                # Forward pass and
                output = model(tokens)
                probs = nn.functional.softmax(
                    output, -1
                )  # Softmax over final dimension for classification
                preds = torch.argmax(probs, -1)
                preds = preds.to(device)
                labels = labels.long()

                # Loss and Optimizer step
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                optimizer.step()

                acc = accuracy_torch(preds, labels, num_classes, device=device)
                f1 = f1_torch(preds, labels, num_classes, device=device)
                train_epoch_f1 += f1
                train_epoch_loss += loss.item()
                train_epoch_acc += acc

                if check:
                    return

            # Normalize training loss and accuracies
            train_epoch_acc = train_epoch_acc / len(iterator.train_loader)
            train_epoch_loss = train_epoch_loss / len(iterator.train_loader)
            train_epoch_f1 = train_epoch_f1 / len(iterator.train_loader)
            print(f"Epoch: {str(iters).zfill(2)}, Train Accuracy: {train_epoch_acc:.4f}, Train Loss: {train_epoch_loss:.4f}, Train F1: {train_epoch_f1:.4f}")

            # Evaluate on validation set
            val_epoch_loss, val_epoch_acc, val_epoch_f1 = evalm(model, iterator, criterion, cust_metrics, custScores, num_classes, val=True, check=check)
            print(f"Epoch: {str(iters).zfill(2)}, Val   Accuracy: {val_epoch_acc:.4f}, Val   Loss: {val_epoch_loss:.4f}, Val   F1: {val_epoch_f1:.4f}")

            if train_epoch_acc < prev_train_acc:
                bad += 1
            prev_train_acc = train_epoch_acc
            iters += 1
        return model, train_epoch_acc, train_epoch_acc

# Train
# model, epoch_loss, epoch_acc = train(lstmMod, dataloader, optimizer, criterion, {}, {}, num_classes, check=False)

# torch.save(model, "models/base.mdl")

model = torch.load("models/base.mdl")

example_ID_list = [("Topic", "Title", "_VB39Jo8mAQ")]
youtube = YCS.YoutubeCommentScraper()
youtube.scrape_comments_from_list(example_ID_list)

with open("YT/Topic/Title.pkl", "rb") as f:
    comments = pickle.load(f)
    print(f"Comments: {comments}")
    for id, text, repCount in comments:
        out, mask = dataloader.transformText(text)
        output = model(out)
        probs = nn.functional.softmax(
            output, -1
        )  # Softmax over final dimension for classification
        preds = torch.argmax(probs, -1)
        print(preds) # 0 is negative, 1 is positive
        break
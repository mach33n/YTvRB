import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np
from math import floor, ceil
from typing import Tuple

class NLPDataset(Dataset):
    # *add should be a list of any additional labels to be used from the dataset
    def __init__(self, file_name, train, label, cols, *add):
        nlp_df = pd.read_csv(file_name, usecols=cols)
        nlp_df = nlp_df.dropna(subset=[train, label])
        nlp_df = nlp_df.reset_index(drop=True)
        self.x_train = nlp_df[train]
        self.y_train = nlp_df[label]
        for label in add:
            self.label = nlp_df[label]

        label_val = nlp_df["overall"]
        self.labels, self.label_idxes, self.label_counts = np.unique(label_val, return_inverse=True, return_counts=True)

        category_str = nlp_df["Category"]
        self.categories, self.category_idxes = np.unique(
            category_str, return_inverse=True
        )

    def get_num_classes(self) -> int:
        """Return the number of classes for classification"""
        return len(self.labels), 1 - (self.label_counts / sum(self.label_counts))

    def get_category_name(self, cat_idx: int) -> str:
        """Return the name of a category from a given category index"""
        return self.categories[cat_idx]

    def get_label_value(self, label_idx: int) -> int:
        """Return the value of the label from a given label index"""
        return self.labels[label_idx]

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx: int) -> Tuple[str, int, int]:
        """
        Returns:
            Tuple[str, int, str]: text review, label index, and category index
        """
        return self.x_train[idx], self.label_idxes[idx], self.category_idxes[idx]


class EADataLoader:
    def __init__(self, batch_size=60, max_seq=128, shuffle=True):
        ## Custom Parameters
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.bos_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2

        # ## Shared tokenization transformation. Could evolve and make this unique to individuals in future##
        self.tokenize = DistilBertTokenizer.from_pretrained('bert-base-uncased')
        self.shuffle = shuffle
        self.seed = 42

    def loadData(
        self, file_name, train, label, cols=["reviewText", "overall", "Category"], *add
    ):
        dset = NLPDataset(file_name, train, label, cols=cols, *add)
        self.num_classes, self.class_weights = dset.get_num_classes()
        dataset_size = len(dset)
        indices = list(range(dataset_size))
        test_split = int(floor(0.85 * dataset_size))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        # Split for Train and Test
        train_indices, test_indices = indices[test_split:], indices[:test_split]
        test_sampler = SubsetRandomSampler(test_indices)

        # Split for Train and Val
        val_split = int(floor(0.6 * len(train_indices)))
        train_indices, val_indices = (
            train_indices[val_split:],
            train_indices[:val_split],
        )
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(
            dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            sampler=train_sampler,
        )
        self.val_loader = DataLoader(
            dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            sampler=val_sampler,
        )
        self.test_loader = DataLoader(
            dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            sampler=test_sampler,
        )

    def collate_fn(self, inp):
        out = list(map(lambda x_new: self.transformText(x_new[0]), inp))
        x_mod = pad_sequence([tup[0] for tup in out])
        x_mask = pad_sequence([tup[1] for tup in out])
        y_mod = torch.Tensor(list(map(lambda x_new: x_new[1], inp)))
        x_mod = torch.transpose(x_mod, 1, 0)
        categories = torch.tensor(list(map(lambda x_new: x_new[2], inp)))
        return x_mod, x_mask, y_mod, categories

    def transformText(self, text):
        try:
            out = self.tokenize(text, return_tensors='pt',  truncation=True, max_length=128, padding="max_length")
        except:
            print(text)
            raise Exception
        masks = out['attention_mask']
        out = out['input_ids']
        out = torch.tensor(out)
        masks = torch.tensor(masks)
        return out, masks
        
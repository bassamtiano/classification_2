import re
import sys
import enum
import pickle
import os

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pytorch_lightning as pl
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from tqdm import tqdm

from transformers import BertTokenizer

class MultiClassPreprocessor():

    def __init__(self, 
                 max_length, 
                 preprocessed_dir, 
                 train_data_dir,
                 test_data_dir,
                 batch_size):
        super(MultiClassPreprocessor, self).__init__()

        self.label2id = {
            'bola': 0,
            'news': 1,
            'bisnis': 2,
            'tekno': 3,
            'otomotif': 4
        }

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.max_length = max_length
        self.preprocessed_dir = preprocessed_dir

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.batch_size = batch_size

    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()
        # Menghilangkan imbuhan
        return self.stemmer.stem(string)

    def load_data(self,):
        with open(self.train_data_dir, "rb") as tdr:
            train_pkl = pickle.load(tdr)
            train = pd.DataFrame({'title': train_pkl[0], 'label': train_pkl[1]})
        with open(self.test_data_dir, "rb") as tsdr:
            test_pkl = pickle.load(tsdr)
            test = pd.DataFrame({'title': test_pkl[0], 'label': test_pkl[1]})
        
        # Mengetahui apa saja label yang ada di dalam dataset
        label_yang_ada = train["label"].drop_duplicates()

        # Konversi dari label text (news) ke label id (1)
        train.label = train.label.map(self.label2id)
        test.label = test.label.map(self.label2id)

        return train, test
    
    def arrange_data(self, data, type):
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []
        for baris, dt in enumerate(tqdm(data.values.tolist())):
            title = self.clean_str(dt[0])
            label = dt[1]


            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1

            tkn = self.tokenizer(text = title,
                                 max_length = self.max_length,
                                 padding = "max_length",
                                 truncation = True)

            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids']) 
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)
            
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)

        if type == "train":

            train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [
                                                                round(len(x_input_ids) * 0.8), 
                                                                len(x_input_ids) - round(len(x_input_ids) * 0.8)
                                                          ])

            # train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [8, 2, 1])


            torch.save(train_tensor_dataset, f"{self.preprocessed_dir}/train.pt")
            torch.save(valid_tensor_dataset, f"{self.preprocessed_dir}/valid.pt")

            return train_tensor_dataset, valid_tensor_dataset
        
        else:
            torch.save(tensor_dataset, f"{self.preprocessed_dir}/test.pt")
            return tensor_dataset

    def preprocessor(self,):
        train, test = self.load_data()

        if not os.path.exists(f"{self.preprocessed_dir}/train.pt") or not os.path.exists(f"{self.preprocessed_dir}/valid.pt"):
            print("Create Train and Validation Dataset")
            train_data, valid_data = self.arrange_data(data = train, type = "train")
        else:
            print("Load Train and Validation Dataset")
            train_data = torch.load(f"{self.preprocessed_dir}/train.pt")
            valid_data = torch.load(f"{self.preprocessed_dir}/valid.pt")

        if not os.path.exists(f"{self.preprocessed_dir}/test.pt"):
            print("Create Test Dataset")
            test_data = self.arrange_data(data = test, type = "test")
        else:
            print("Load Test Dataset")
            test_data = torch.load(f"{self.preprocessed_dir}/test.pt")

        return train_data, valid_data, test_data


    def preprocessor_manual(self):
        train_data, valid_data, test_data = self.preprocessor()


        train_sampler = RandomSampler(train_data)
        valid_sampler = SequentialSampler(valid_data)
        test_sampler = SequentialSampler(test_data)

        train_dataset = DataLoader(
            dataset = train_data,
            batch_size = self.batch_size,
            sampler = train_sampler,
            num_workers = 3
        )

        valid_dataset = DataLoader(
            dataset = valid_data,
            batch_size = self.batch_size,
            sampler = valid_sampler,
            num_workers = 3
        )

        test_dataset = DataLoader(
            dataset = test_data,
            batch_size = self.batch_size,
            sampler = test_sampler,
            num_workers = 3
        )

        return train_dataset, valid_dataset, test_dataset

    
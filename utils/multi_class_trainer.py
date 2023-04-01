import sys
import random
from statistics import mean

import torch
import torch.nn as nn

from model.multi_class_model import MultiClassModel

from sklearn.metrics import f1_score
from tqdm import tqdm

class MultiClassTrainer(object):
    def __init__(self,
                 dropout,
                 lr,
                 max_epoch,
                 device,
                 n_class):
        super(MultiClassTrainer, self).__init__()
        
        self.lr = lr
        self.max_epoch = max_epoch
        # Control Random
        self.random_seed_generator()
        
        # Inisialisasi Model
        self.model = MultiClassModel(n_out = n_class, dropout = dropout)
        
        # Inisialisasi hitung loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Meletakkan / Alokasi model ke memori apa (GPU / CPU)
        self.device = device
        self.model.to(self.device)
        
        # Inisialisasi optimizer 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor = 0.5, total_iters = 3)
    
    def random_seed_generator(self):
        random.seed(69)
        torch.manual_seed(69)
        torch.cuda.manual_seed(69)
        torch.cuda.manual_seed_all(69)
        torch.backends.cudnn.deterministic = True

    def train_step(self):
        
        self.model.train()
        self.model.zero_grad()
        
        scores = {"loss": [],
                  "f1_micro": []}
        
        progress_train = tqdm(self.train_dataset)
        
        for batch in progress_train:
            x_input_ids, x_token_type_ids, x_attention_mask, y = batch
            
            # arange text dan label ke memory GPU / CPU
            x_input_ids = x_input_ids.to(self.device)
            x_token_type_ids = x_token_type_ids.to(self.device)
            x_attention_mask = x_attention_mask.to(self.device)
            y = y.to(self.device)
            
            out = self.model(x_input_ids = x_input_ids)
            
            
            loss = self.criterion(out, target = y.float())
            
            # 0,0,0,1 = 3
            pred = out.argmax(1).cpu()
            true = y.argmax(1).cpu()
            
            f1_micro = round(f1_score(pred, true, average="micro"), 2)
            
            scores["loss"].append(loss.item())
            scores["f1_micro"].append(f1_micro)
            
            loss.backward()
            self.optimizer.step()
            
        self.scheduler.step()
        
        return {
            "loss": round(mean(scores["loss"]), 2),
            "f1_micro": round(mean(scores["f1_micro"]), 2)
        }
            
    def validation_step(self):
        with torch.no_grad():
            
            scores = {"loss": [],
                      "f1_scores": []}
            
            self.model.eval()
            
            progress_val = tqdm(self.validation_dataset)
            
            for batch in progress_val:
                x_input_ids, _, _, y = batch
                
                # arange text dan label ke memory GPU / CPU
                x_input_ids = x_input_ids.to(self.device)
                y = y.to(self.device)
                
                # Proses input di model
                out = self.model(x_input_ids = x_input_ids)
                # Loss training pada model
                loss = self.criterion(out, target = y.float())
                
                pred = out.argmax(1).cpu()
                true = y.argmax(1).cpu()
                
                f1_micro = round(f1_score(pred, true, average="micro"), 2)
                
                scores["loss"].append(loss.item())
                scores["f1_micro"].append(f1_micro)
            
            return {
                "loss": round(mean(scores["loss"]), 2),
                "f1_micro": round(mean(scores["f1_micro"]), 2)
            }
        
    def test_step(self):
        with torch.no_grad():
            
            scores = {"loss": [],
                      "f1_scores": []}
            
            self.model.eval()
            
            progress_val = tqdm(self.test_dataset)
            
            for batch in progress_val:
                x_input_ids, _, _, y = batch
                
                # arange text dan label ke memory GPU / CPU
                x_input_ids = x_input_ids.to(self.device)
                y = y.to(self.device)
                
                # Proses input di model
                out = self.model(x_input_ids = x_input_ids)
                # Loss training pada model
                loss = self.criterion(out, target = y.float())
                
                pred = out.argmax(1).cpu()
                true = y.argmax(1).cpu()
                
                f1_micro = round(f1_score(pred, true, average="micro"), 2)
                
                scores["loss"].append(loss.item())
                scores["f1_micro"].append(f1_micro)
            
            return {
                "loss": round(mean(scores["loss"]), 2),
                "f1_micro": round(mean(scores["f1_micro"]), 2)
            }
    
    def trainer(self, train_dataset, validation_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        print(self.train_dataset)
        
        for epoch in range(self.max_epoch):
            print("Epoch = ", epoch)
            
            # Trainig Step
            self.train_step()
            print("test")
            validation_scores = self.validation_step()
            
        #     print(validation_scores)
            
        # test_scores = self.test_step()
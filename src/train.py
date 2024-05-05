import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.neural_model import NeuralNet
from src.intent_classes import Training_Data_XY
from src.AI import ChatAI

from datetime import datetime
import time

from uuid import uuid1 as getuuid



class ChatDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def evaluate_intent(intents: list[dict[str, str | list[str]]]):
    Tags = []
    Patterns = []
    for intent in intents:
        tag = intent["tag"].lower()
        if tag in Tags:
            raise ValueError(f"Duplicate tag name {tag}")
        for pattern in intent["patterns"]:
            if pattern.lower() in Patterns:
                raise ValueError(f"Duplicate pattern name {pattern} at tag: {tag}")
            Patterns.append(pattern.lower())
        Tags.append(tag)

FOLDER_Path = "models/"
EXTENSION = ".model"

def generate_filename():
    # date, time = str(datetime.now()).replace(":", "-").split(" ")
    filename = f"data-{int(time.time())}-id-{getuuid()}"
    return FOLDER_Path + filename + EXTENSION


class Trainer:
    status = "initialize"
    
    NUM_EPOCHS = 1000
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    @property
    def input_size(self) -> int:
        return len(self.X_train[0])
    
    @property
    def output_size(self) -> int:
        return len(self.all_tags)
    
    @property
    def dataset(self):
        return ChatDataset(self.X_train, self.Y_train)
    
    @property
    def train_loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)
        
    def __init__(self, intents: list[dict[str, str | list[str]]]):
        evaluate_intent(intents)
        training_data = Training_Data_XY(intents)
        self.X_train = training_data.X_train
        self.Y_train = training_data.Y_train
        self.all_words = training_data.Stemmed_words
        self.all_tags = training_data.All_tags
        self.intents = intents
        
        self.model = NeuralNet(self.input_size, self.HIDDEN_SIZE, self.output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        
    def start_train(self, print_output = False):
        # Train the model
        for epoch in range(self.NUM_EPOCHS):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if (epoch+1) % 100 == 0:
                self.status = f'Epoch [{epoch+1}/{self.NUM_EPOCHS}], Loss: {loss.item():.4f}'
                if (print_output):
                    print(self.status)
        self.status = f'Final loss: {loss.item():.4f}'
        if (print_output):
            print(self.status)
        
    def get_model(self) -> ChatAI:
        return ChatAI(
            self.input_size,
            self.HIDDEN_SIZE,
            self.output_size,
            self.model.state_dict(),
            self.intents
        )
                
    def save_model(self):
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.HIDDEN_SIZE,
            "output_size": self.output_size,
            "intents": self.intents,
            }
        filename = generate_filename()
        torch.save(data, filename)
        print(f'training complete. file saved to {filename}')

                
if __name__ == '__main__':
    # from src.intent import Intents
    
    # train = Trainer(Intents)
    # train.start_train()
    # train.save_model()
    pass
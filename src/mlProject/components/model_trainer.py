from mlProject.entity.config_entity import ModelTrainerConfig
from copy import deepcopy as dc
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim
from torch.utils.data import DataLoader
import joblib



from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)
        def     __getitem__(self, i):
            return self.X[i], self.y[i]
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers=2, device="cpu"):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.device = device

        def forward(self, x):
            bacth_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, bacth_size, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_stacked_layers, bacth_size, self.hidden_size).to(self.device )

            # Forward pass through LSTM
            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out, hn, cn

    def validate_one_epoch(self, model, test_loader, loss_function, device):
        model.train(False)
        running_loss = 0

        for _, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                output,_,_ = model(x_batch)

                loss = loss_function(output, y_batch)
                running_loss+=loss

        average_loss_across_batches = running_loss/len(test_loader)
        logger.info("Val Loss {0:.3f}".format(average_loss_across_batches))


    def training(self, model,
                    train_loader,
                    test_loader,
                    num_epochs, 
                    loss_function, 
                    optimizer, scheduler,
                    validate_one_epoch_function, device):
        model.train(True)
        for epoch in range(1, num_epochs+1):
            running_loss = 0.0
            for _, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                output,_ ,_= model(x_batch)        

                loss = loss_function(output, y_batch)
                running_loss+=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_loss_across_batches = running_loss/len(train_loader)

            if epoch % 10 == 0:
                logger.info(f"Epoch :{epoch}, Loss: {average_loss_across_batches}")
                validate_one_epoch_function(model, test_loader, loss_function, device)
            running_loss = 0.0
            scheduler.step(average_loss_across_batches.detach().cpu().item())
            
        return model


    def train(self):
        train_data = pd.read_csv(self.config.train_data_path, index_col="time")
        test_data = pd.read_csv(self.config.test_data_path, index_col="time")
        
        classification_target_column = list(self.config.target_column_classification.keys())[0]
        regression_target_column = list(self.config.target_column.keys())[0]
        targets_columns = [classification_target_column, regression_target_column]

        # Features
        X_train_classification = train_data.drop(columns=targets_columns).values
        X_test_classification = test_data.drop(columns=targets_columns).values

        # Target
        classification_y_train  = train_data[classification_target_column]
        classification_y_test = test_data[classification_target_column]


        # Scale features
        scaler = StandardScaler()
        X_train_classification_scaled = scaler.fit_transform(X_train_classification)
        X_test_classification_scaled = scaler.transform(X_test_classification)

        # Train logistic regression
        lr = LogisticRegression(**self.config.classification_model_params)
        lr.fit(X_train_classification_scaled, classification_y_train)

        # Save scaler and model
        joblib.dump(scaler, os.path.join(self.config.root_dir, self.config.scaler_name))
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.classification_model_name))


        ## Lstm model
        X_train= dc(np.flip(X_train_classification_scaled, axis=1))
        X_test= dc(np.flip(X_test_classification_scaled, axis=1))
        vars_dim = X_train.shape[1]


        X_train = X_train.reshape((-1, vars_dim, 1))
        X_test = X_test.reshape((-1, vars_dim, 1))

        
        y_train =  train_data[regression_target_column].values
        y_test =  test_data[regression_target_column].values
        y_train =y_train.reshape((-1, 1))
        y_test =y_test.reshape((-1, 1))
        
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()

        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()
        
        train_dataset = self.TimeSeriesDataset(X_train, y_train)
        test_dataset = self.TimeSeriesDataset(X_test, y_test)
        batch_size = self.config.batch_size
        
        train_loader =  DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_loader =  DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        input_size = self.config.lstm_model_params.input_size
        hidden_size = self.config.lstm_model_params.hidden_size
        num_stacked_layers = self.config.lstm_model_params.num_stacked_layers
        model = self.LSTM(input_size, hidden_size, num_stacked_layers, device=device)
        
        model.to(device)

        
        leraning_rate = float(self.config.lstm_model_params.learning_rate)
        num_epochs = self.config.lstm_model_params.num_epochs
        optimizer = optim.Adam(model.parameters(), lr = leraning_rate)
        loss_function = nn.MSELoss()
        factor, patience = float(self.config.lstm_model_params.scheduler_factor), int(self.config.lstm_model_params.scheduler_patience)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)

        trained_model = self.training(
                                    model,
                                    train_loader,
                                    test_loader,
                                    num_epochs, 
                                    loss_function, 
                                    optimizer, scheduler,
                                    self.validate_one_epoch,
                                    device=device)

        torch.save(trained_model,  os.path.join(self.config.root_dir, self.config.lstm_model_name))
                
        logger.info("Training completed and models saved.")



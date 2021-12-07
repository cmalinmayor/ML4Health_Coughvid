import numpy as np
import pandas as pd
from datetime import datetime

from coughvid.pytorch import CoswaraDataset, SubsetWeightedRandomSampler, compute_weights
from .evaluate_model import Evaluator
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import resnet50, resnet18
import os
import logging

from sklearn.model_selection import train_test_split
import copy

from leaf_audio_pytorch import frontend, initializers
from coughvid.audio_processing import feature_extraction

import wandb

logger = logging.getLogger(__name__)


class CoswaraTrainer:
    def __init__(self, data_dir, batch_size=1, num_workers=1, model_dir='trained_models', leaf=False, augmentation=False, normalization=True, energy_filter=False, name="coswara_model"):
        self.data_dir = data_dir  # './data/coswara/'
        self.metadata_file = 'filtered_data.csv'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_dir = model_dir
        self.leaf=leaf
        self.augmentation=augmentation
        self.normalization=normalization
        self.energy_filter = energy_filter
        self.name = name
        os.makedirs(model_dir, exist_ok=True)

    def load_model(self, model_type='resnet18',
                   optim=torch.optim.Adam, loss=torch.nn.BCELoss):
        # load model and change output shape for binary prediction
        if model_type == 'resnet18':
            model = resnet18()
            features_in_final_layer = 512
        elif model_type == 'resnet50':
            model = resnet50()
            features_in_final_layer = 2048
        else:
            raise NotImplementedError(
                    "Implemented model types: resnet18, resnet50")

        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.10, inplace=True),
            torch.nn.Linear(
                in_features=features_in_final_layer,
                out_features=1
            ),
            torch.nn.Sigmoid()
        )
        
        class my_leaf(torch.nn.Module):
            def __init__(self):
                super(my_leaf, self).__init__()
                n_filters = 40
                window_stride = 23.22
                sample_rate = 44100
                self.custom_leaf = frontend.Leaf(n_filters=n_filters,
                                   window_stride=window_stride,
                                   sample_rate=sample_rate)
            def forward(self, X):
                out = self.custom_leaf(X)
                out = out[None,...].double()
                return out
            

        model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        
        if self.leaf:
            model = torch.nn.Sequential(my_leaf(), model)
            
        optimizer = optim(model.parameters())
        criterion = loss()
        model.double()
        return model, optimizer, criterion

    def get_dataloaders(self):
        full_dataset = CoswaraDataset(
                self.data_dir,
                self.metadata_file,
                normalization=self.normalization,
                get_features=True,
                get_leaf=self.leaf,
                augmentation=self.augmentation,
                energy_filter=self.energy_filter
                )
        dataframe = full_dataset.dataframe
        minority_class_count = len(dataframe[dataframe['covid_status'] == 1])
        samples_per_epoch = minority_class_count*2
        print(f'{samples_per_epoch} samples per_epoch.')

        # split data into training and test samples
        train_indices, test_indices = train_test_split(
                np.arange(0, len(full_dataset)-1),
                test_size=0.25,
                random_state=4789)
        labels = list(dataframe['covid_status'])
        train_weights = compute_weights(labels, train_indices)
        train_sampler = SubsetWeightedRandomSampler(
                train_indices, train_weights, samples_per_epoch)

        train_loader = DataLoader(full_dataset,
                                  num_workers=self.num_workers,
                                  sampler=train_sampler
                                  )

        test_loader = DataLoader(full_dataset,
                                 num_workers=self.num_workers,
                                 sampler=SubsetRandomSampler(test_indices)
                                 )

        dataloaders = {
            "train": train_loader,
            "test": test_loader
        }
        return dataloaders
    
    def leaf_test_step(self, model, dataloader):
        model.eval()
        all_labels = []
        all_predictions = []
        for j, batch in enumerate(dataloader):
            X, labels = batch
            X = X.flatten()
            X = X[None,None,...].float()
            
            all_labels.extend(list(labels.cpu().numpy()))
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                y = model(X)
                all_predictions.extend(list(y.cpu().numpy()))
        return all_labels, all_predictions

    def test_step(self, model, dataloader):
        if self.leaf:
            return self.leaf_test_step(model, dataloader)
        model.eval()
        all_labels = []
        all_predictions = []
        for j, batch in enumerate(dataloader):
            X, labels = batch
            all_labels.extend(list(labels.cpu().numpy()))
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                y = model(X[None, ...].double())
                all_predictions.extend(list(y.cpu().numpy()))
        return all_labels, all_predictions

    def training_step(
            self, model, dataloader, optimizer, criterion,
            use_wandb=False):
        model.train()
        samples = 0
        loss_sum = 0
        correct_sum = 0
        for j, batch in enumerate(dataloader):
            X, labels = batch
            
            #scripts for leaf
            if self.leaf:
                X = X.flatten()
                X = X[None,None,...].float()
            
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if self.leaf:
                    y = model(X)
                else:
                    y = model(X[None, ...].double())
                loss = criterion(
                    y,
                    labels[None, ...].double()
                )

                loss.backward()
                optimizer.step()

                # Multiply by batch size as loss is the mean loss of the samples in the batch
                loss_sum += loss.item() * X.shape[0]
                samples += X.shape[0]
                num_corrects = torch.sum(
                        (y >= 0.5).float() == labels[0].float())
                correct_sum += num_corrects

                # Print batch statistics every 50 batches
                if j % 50 == 49:
                    if use_wandb:
                        wandb.log({'Train Loss': float(loss_sum) / float(samples),
                                   'Train Accuracy': float(correct_sum) / float(samples)})

                    print("{} - loss: {}, acc: {}".format(
                        j + 1,
                        float(loss_sum) / float(samples),
                        float(correct_sum) / float(samples)
                    ))
        return loss_sum, samples, correct_sum

    def train_model(
            self,
            model_type='resnet18',
            num_epochs=50,
            use_wandb=False):

        model, optimizer, criterion = self.load_model(model_type)
        dataloaders = self.get_dataloaders()
        # RESNET training code adapted from
        # https://www.kaggle.com/gxkok21/resnet50-with-pytorch
        if torch.cuda.is_available():
            model = model.cuda()
            print('Using GPU.')
        else:
            print('NOT Using GPU. :(')

        if use_wandb:
            wandb.init()
            wandb.watch(model)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        evaluator = Evaluator(model_name=self.name, path_to_logs=self.name)

        # create dataframe for storing stats
        csv_df = pd.DataFrame(columns=[
            'Epoch',
            'Training Accuracy',
            'Test Accuracy',
            'Training Loss',
            'Test Loss'
            ])

        str_date_time = datetime.now().strftime("%d%m%Y%H%M%S")

        for i in range(num_epochs):
            epoch = i + 1
            # train step
            loss_sum, samples, correct_sum = self.training_step(
                    model, dataloaders['train'],
                    optimizer, criterion,
                    use_wandb=use_wandb)

            # Print epoch statistics
            epoch_acc = float(correct_sum) / float(samples)
            epoch_loss = float(loss_sum) / float(samples)
            logger.debug("epoch: {} - train loss: {}, train acc: {}".format(
                    i + 1, epoch_loss, epoch_acc))

            #logging
            csv_df.loc[i,'Epoch'] = i+1
            csv_df.loc[i,'Training Accuracy' ] = epoch_acc
            csv_df.loc[i,'Training Loss' ] = epoch_loss

            # test step
            labels, predictions = self.test_step(model, dataloaders['test'])
            #logging
            csv_df.loc[i,'Epoch'] = i+1
            csv_df.loc[i,'Test Accuracy' ] = epoch_acc
            csv_df.loc[i,'Test Loss' ] = epoch_loss

            evaluator.evaluate_epoch(epoch, labels, predictions, epoch_acc)

            # Deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                filename = os.path.join(self.model_dir,
                                        f"resnet18_coswara_epoch_{i}_{str_date_time}.pth")
                logger.info(f"Saving model to {filename}")
                torch.save(best_model_wts, filename)

        # finish logging
        name = model_type + str_date_time + '.csv'
        csvfile = open(name, 'w')
        csv_df.to_csv(csvfile, index=False)
        csvfile.close()

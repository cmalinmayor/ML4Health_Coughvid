import numpy as np
from datetime import datetime

from coughvid.pytorch import CoswaraEmbeddingDataset, SubsetWeightedRandomSampler, compute_weights
from .evaluate_model import Evaluator
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import logging

from sklearn.model_selection import train_test_split
import copy

import wandb

logger = logging.getLogger(__name__)

0.000005
class LR(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        # torch.nn.Sigmoid()

    def forward(self, x):
        pred = self.linear(x)
        return pred


class CoswaraEmbeddingTrainer:
    def __init__(self, data_dir, batch_size=1, num_workers=1, model_dir='trained_models', name="embedding_model", unit_sec=6.0, feature_d=512, lr=0.00001):
        self.data_dir = data_dir  # './data/coswara/'
        self.metadata_file = 'filtered_data.csv'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_dir = model_dir
        self.name = name
        self.unit_sec = unit_sec
        self.feature_d = feature_d
        self.lr = lr
        os.makedirs(model_dir, exist_ok=True)

    def load_model(self, model_type='linear',
                   optim=torch.optim.SGD, loss=torch.nn.BCEWithLogitsLoss,
                   lr=0.00001):
        # load model and change output shape for binary prediction
        if model_type == 'linear':
            model = LR(self.feature_d, 1)
        else:
            raise NotImplementedError(
                    "Implemented model types: linear")

        optimizer = optim(model.parameters(), lr=lr)
        criterion = loss()
        model.double()
        return model, optimizer, criterion

    def get_dataloaders(self):
        full_dataset = CoswaraEmbeddingDataset(
                self.data_dir,
                self.metadata_file,
                unit_sec = self.unit_sec,
                feature_d = self.feature_d
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
                                  sampler=train_sampler,
                                  batch_size=self.batch_size
                                  )

        test_loader = DataLoader(full_dataset,
                                 num_workers=self.num_workers,
                                 sampler=SubsetRandomSampler(test_indices),
                                 batch_size=self.batch_size
                                 )

        dataloaders = {
            "train": train_loader,
            "test": test_loader
        }
        return dataloaders

    def test_step(self, model, dataloader):
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
                y = model(X.double())
                y = torch.nn.functional.sigmoid(y)
                all_predictions.extend(list(y.cpu().numpy()))
        print(all_predictions)
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
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()
                labels = torch.unsqueeze(labels, -1)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                y = model(X.double())
                loss = criterion(
                    y,
                    labels.double()
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
            model_type='linear',
            num_epochs=50,
            use_wandb=False):

        model, optimizer, criterion = self.load_model(model_type, lr=self.lr)
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

        evaluator = Evaluator(model_name=self.name,
                              path_to_logs=self.name,
                              threshold=0.5)

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


            # test step
            labels, predictions = self.test_step(model, dataloaders['test'])
            evaluator.evaluate_epoch(epoch, labels, predictions, epoch_acc)

            # Deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                filename = os.path.join(self.model_dir,
                                        f"linear_coswara_epoch_{i}_{str_date_time}.pth")
                logger.info(f"Saving model to {filename}")
                torch.save(best_model_wts, filename)

import numpy as np

from coughvid.pytorch.coswara_dataset import CoswaraDataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import resnet50, resnet18

from sklearn.model_selection import train_test_split
import copy

import wandb


class CoswaraTrainer:
    def __init__(self, data_dir, batch_size=1, num_workers=1):
        self.data_dir = data_dir  # './data/coswara/'
        self.metadata_file = 'filtered_data.csv'
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        optimizer = optim(model.parameters())
        criterion = loss()
        model.double()
        return model, optimizer, criterion

    def get_dataloaders(self, leaf=False, augmentation=False):
        full_dataset = CoswaraDataset(
                self.data_dir, self.metadata_file, get_features=True)
        dataframe = full_dataset.dataframe
        minority_class_count = len(dataframe[dataframe['covid_status'] == 1])
        print(f'{minority_class_count} samples in the minority class.')
        # TODO: don't limit number of samples from majority class
        sample_dataset = CoswaraDataset(
                self.data_dir, 'filtered_data.csv',
                get_features=True, samples_per_class=minority_class_count)

        # split data into training and test samples
        train_indices, test_indices = train_test_split(
                np.arange(0, len(sample_dataset)-1), test_size=0.25)
        train_loader = DataLoader(sample_dataset,
                                  num_workers=self.num_workers,
                                  sampler=SubsetRandomSampler(train_indices)
                                  )

        test_loader = DataLoader(sample_dataset,
                                 num_workers=self.num_workers,
                                 sampler=SubsetRandomSampler(test_indices)
                                 )

        dataloaders = {
            "train": train_loader,
            "test": test_loader
        }
        return dataloaders

    def training_step(
            self, model, dataloader, optimizer, criterion,
            phase='train', use_wandb=True):
        samples = 0
        loss_sum = 0
        correct_sum = 0
        for j, batch in enumerate(dataloader):
            X, labels = batch
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                y = model(X[None, ...].double())
                loss = criterion(
                    y,
                    labels[None, ...].double()
                )

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # Multiply by batch size as loss is the mean loss of the samples in the batch
                loss_sum += loss.item() * X.shape[0]
                samples += X.shape[0]
                num_corrects = torch.sum(
                        (y >= 0.5).float() == labels[0].float())
                correct_sum += num_corrects

                # Print batch statistics every 50 batches
                if j % 50 == 49 and phase == "train":
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
            use_wandb=True):
        dataloaders = self.get_dataloaders()
        model, optimizer, criterion = self.load_model(model_type)

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

        for i in range(num_epochs):
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                loss_sum, samples, correct_sum = self.training_step(
                        model, dataloaders[phase],
                        optimizer, criterion,
                        phase=phase, use_wandb=use_wandb)

                # Print epoch statistics
                epoch_acc = float(correct_sum) / float(samples)
                epoch_loss = float(loss_sum) / float(samples)
                print("epoch: {} - {} loss: {}, {} acc: {}".format(
                        i + 1, phase, epoch_loss, phase, epoch_acc))

                # Deep copy the model
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, "resnet18_coswara2.pth")

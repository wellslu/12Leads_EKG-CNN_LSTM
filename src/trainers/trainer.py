from abc import ABCMeta, abstractmethod
import os
import mlconfig
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, device, model, optimizer, scheduler, train_loader, test_loader, num_epochs):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.crition = nn.L1Loss()
        # self.crition = nn.CrossEntropyLoss()

        self.epoch = 1
        self.best_loss = 999

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
#             self.train_loader.dataset.random_sample()
#             self.test_loader.dataset.random_sample()
            train_loss = self.train()
            test_loss = self.evaluate()
            self.scheduler.step()
            self.save_checkpoint('checkpoint.pth')

            metrics = dict(train_loss=train_loss.value,
#                            train_acc=train_acc.value,
                           test_loss=test_loss.value,)
#                            test_acc=test_acc.value)
            mlflow.log_metrics(metrics, step=self.epoch)

            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {},'.format(train_loss)
            format_string += 'test loss: {}, '.format(test_loss)
            format_string += 'best test loss: {}.'.format(self.best_loss)
#             format_string += 'train loss: {}, train acc: {}, '.format(train_loss, train_acc)
#             format_string += 'test loss: {}, test acc: {}, '.format(test_loss, test_acc)
#             format_string += 'best test acc: {}.'.format(self.best_acc)
            tqdm.write(format_string)


    def train(self):
        self.model.train()

        train_loss = Average()
#         train_acc = Accuracy()

        for x, bef, y in tqdm(self.train_loader):
            x = x.to(self.device)
            bef = torch.Tensor(bef).type(torch.FloatTensor).to(self.device)
            y = torch.Tensor(y).type(torch.FloatTensor).to(self.device)

            output = self.model(x, bef)
            loss = self.crition(output, y.reshape(-1, 1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
#             train_acc.update(output, y)

        return train_loss

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
#         test_acc = Accuracy()

        with torch.no_grad():
            for x, bef, y in tqdm(self.test_loader):
                x = x.to(self.device)
                bef = bef = torch.Tensor(bef).type(torch.FloatTensor).to(self.device)
                y = torch.Tensor(y).type(torch.FloatTensor).to(self.device)

                output = self.model(x, bef)
                loss = self.crition(output, y.reshape(-1, 1))

                test_loss.update(loss.item(), number=x.size(0))
#                 test_acc.update(output, y)

        if test_loss.value < self.best_loss:
            self.best_loss = test_loss.value
            self.save_checkpoint('best.pth')
            torch.save(self.model, 'best.pt')

        return test_loss

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

import argparse

import mlconfig
import mlflow
import numpy as np
import torch

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    mlflow.log_artifact(args.config)
    mlflow.log_params(config.flat())

    manual_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = 'cpu'
    model = config.model().to(device)
    model.load_state_dict(torch.load('./9726.pth')['model'])
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    train_loader = config.dataset(file='./data/all.csv', train=True)
    test_loader = config.dataset(file='./data/validation.csv', train=False)

    trainer = config.trainer(device, model, optimizer, scheduler, train_loader, test_loader)

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()


if __name__ == '__main__':
    main()

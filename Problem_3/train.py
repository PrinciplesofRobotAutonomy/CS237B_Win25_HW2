#!/usr/bin/env python

import argparse
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

from model import AccelerationPredictionNetwork, BaselineNetwork, loss
import utils

SIZE_BATCH = 32

# Feel free to experiment with these parameters!
LEARNING_RATE = 1e-5
NUM_EPOCHS = 30

PATH_CHECKPOINT = os.path.join('trained_models', 'cp-{epoch:03d}.pt')
DIR_MODEL = 'trained_models'

def train_model(model, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if isinstance(model, BaselineNetwork):
            images, targets = batch[0], batch[1]
            outputs = model(images)
        else:
            images, angles, targets = batch
            outputs = model(images, angles)

        targets = targets.reshape(-1, 1)
        loss_value = loss(targets, outputs)
        loss_value.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss_value.item(), epoch * len(train_loader) + batch_idx)
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_value.item():.6f}')

def test_model(model, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(model, BaselineNetwork):
                images, targets = batch[0], batch[1]
                outputs = model(images)
            else:
                images, angles, targets = batch
                outputs = model(images, angles)
            test_loss += loss(targets.reshape(-1, 1), outputs).item()
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Loss/test', test_loss, epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    args = parser.parse_args()

    # Create directory if it doesn't exist
    os.makedirs(DIR_MODEL, exist_ok=True)

    # Load dataset
    ramp_dataset_path = os.path.join('phys101', 'scenarios', 'ramp')
    train_loader, test_loader = utils.load_dataset(ramp_dataset_path,
                                                   ramp_surface=1,  # Choose ramp surface in experiments (1 or 2)
                                                   size_batch=SIZE_BATCH)

    # Build model
    if args.baseline:
        model = BaselineNetwork()
        path_model = os.path.join(DIR_MODEL, "trained_baseline.pt")
    else:
        model = AccelerationPredictionNetwork()
        path_model = os.path.join(DIR_MODEL, "trained.pt")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir='train_logs')

    # Train model
    for epoch in range(1, NUM_EPOCHS + 1):
        train_model(model, train_loader, optimizer, epoch, writer)
        test_model(model, test_loader, writer, epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), PATH_CHECKPOINT.format(epoch=epoch))

    torch.save(model.state_dict(), path_model)
    writer.close()

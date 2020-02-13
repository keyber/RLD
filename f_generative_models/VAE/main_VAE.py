import torch
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split
import shutil
import os
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from VAE import VAE, loss_function
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 8
H_DIM = 256
Z_DIM = 16
LOSS_PATH = "losses/{}".format(Z_DIM)
IMAGES_PATH = "images/{}".format(Z_DIM)
CHECKPOINT_PATH = "checkpoint/h={}_z={}".format(H_DIM, Z_DIM)

def save_model(epoch, network, optimizer, last_train_loss, last_test_loss, opt_reached, checkpoint, iteration):
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_train_loss': last_train_loss,
        'last_test_loss': last_test_loss,
        'opt_reached': opt_reached
    }, '{}.pth'.format(checkpoint))

def create_grid(network, im, writer=None, epoch=None):
    with torch.no_grad():
        reconstructed_im, _, _ = network(im.view((im.shape[0], -1)))
        reconstructed_im = reconstructed_im.reshape(im.shape)
        im = im.repeat(1, 3, 1, 1)
        reconstructed_im = reconstructed_im.repeat(1, 3, 1, 1)

        images = torch.cat((im, reconstructed_im), 0)
        grid_img = make_grid(images, nrow=len(im))

    if writer:
        assert epoch is not None
        writer.add_image('Epoch {}'.format(epoch),
                     grid_img, epoch)
    else:
        return grid_img

def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_trainset = datasets.MNIST(root='data', train=True, download=True,
                                    transform=transform)
    mnist_trainset, mnist_valset = random_split(mnist_trainset,
                                                (int(0.9 * len(mnist_trainset)),
                                                 int(0.1 * len(mnist_trainset))))

    mnist_testset = datasets.MNIST(root='data', download=True, transform=transform)

    # uncomment to train on a fraction of the dataset
    train_subset = torch.randint(0, len(mnist_trainset), (10000,))
    val_subset = torch.randint(0, len(mnist_valset), (1000,))

    train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    ### uncomment the line below and comment the line above to train on a fraction of the ds
                                               # sampler=SubsetRandomSampler(train_subset))
    valid_loader = torch.utils.data.DataLoader(mnist_valset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    ### uncomment the line below and comment the line above to train on a fraction of the ds
                                               #sampler=SubsetRandomSampler(val_subset))

    # test_subset = torch.randint(0, len(mnist_testset), (1000,))
    test_loader = torch.utils.data.DataLoader(mnist_testset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
                                              # sampler=SubsetRandomSampler(test_subset))

    return train_loader, valid_loader, test_loader

def evaluate(network, data_loader, device, writer, epoch):
    network.eval()
    losses = []

    # log images
    im, _ = next(iter(data_loader))
    im = im.to(device)
    create_grid(network, im, writer, epoch)

    with torch.no_grad():
        for step, (batch_x, _) in enumerate(data_loader):
            batch_x = batch_x.view((batch_x.shape[0],-1)).to(device)
            batch_x_hat, mu, log_var = network(batch_x)
            loss = loss_function(batch_x_hat, batch_x, mu, log_var)

            losses.append(loss.item())

    return np.mean(losses)

def train_loop(network, optimizer, train, valid, device, writers, max_epochs=100, eps=1e-4):
    opt_reached = False
    mean_train_loss = None
    mean_test_loss = None
    iteration = 0

    best_loss = np.inf
    best_params = None

    network.train()
    start = time.time()

    t = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        t.update(1)
        train_batch_losses = []
        for step, (batch_x, _) in enumerate(train):
            optimizer.zero_grad()
            iteration += 1

            # (batch_size, w*h)
            batch_x = batch_x.view((batch_x.shape[0], -1)).to(device)

            batch_x_hat, mu, log_var = network(batch_x)
            loss = loss_function(batch_x_hat, batch_x, mu, log_var)
            train_batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        mean_train_loss = np.mean(train_batch_losses)
        mean_test_loss = evaluate(network, valid, device, writers['images'], epoch)

        writers['loss'].add_scalars('loss', {'train': mean_train_loss, 'test': mean_test_loss}, epoch)

        if mean_test_loss < best_loss:
            best_params = network.state_dict()
            best_loss = mean_test_loss

            if mean_train_loss < eps:
                opt_reached = True

            save_model(epoch, network, optimizer, mean_train_loss, mean_test_loss, opt_reached, CHECKPOINT_PATH, iteration)

        if opt_reached:
            break

    stop = time.time()
    if opt_reached:
        print("Optimum reached in {} ({}s) iterations (training loss = {}, "
              "test loss = {})".format(iteration, stop - start, mean_train_loss,
                                       mean_test_loss))
    else:
        print("Optimum not reached in {} ({}s) iterations (training loss = {}, "
              "test loss = {})".format(iteration, stop - start, mean_train_loss,
                                       mean_test_loss))

    return best_loss, best_params


def train(input_dim, train_loader, valid_loader, device, writers):
    vae = VAE(input_dim=input_dim, h_dim=H_DIM, z_dim=Z_DIM)
    optimizer = optim.Adam(vae.parameters())

    _, _ = train_loop(vae, optimizer, train_loader, valid_loader, device, writers, max_epochs=10)

def test(input_dim, test_loader, device, writer):
    vae = VAE(input_dim=input_dim, h_dim=H_DIM, z_dim=Z_DIM)
    checkpoint = torch.load(CHECKPOINT_PATH + '.pth')
    vae.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    validation_loss = checkpoint['last_test_loss']
    print(validation_loss)
    print(epoch)

    im, _ = next(iter(test_loader))
    im = im.to(device)

    create_grid(vae, im, writer=writer, epoch=0)

def main():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    train_loader, valid_loader, test_loader = get_dataloaders()
    inputs, _ = next(iter(train_loader))
    _, _, width, height = inputs.shape

    writer_loss = SummaryWriter(LOSS_PATH)
    writer_images = SummaryWriter(IMAGES_PATH + '/train')
    writer_test = SummaryWriter(IMAGES_PATH + '/test')
    writers = {'loss': writer_loss, 'images': writer_images}

    # train(width * height, train_loader, valid_loader, device, writers)
    test(width * height, test_loader, device, writer_test)

    writer_loss.close()
    writer_images.close()


def main_visualisation():
    train_loader, valid_loader, test_loader = get_dataloaders()
    inputs, _ = next(iter(train_loader))
    _, _, width, height = inputs.shape

    vae = VAE(input_dim=width * height, h_dim=H_DIM, z_dim=Z_DIM)
    checkpoint = torch.load(CHECKPOINT_PATH + '.pth')
    vae.load_state_dict(checkpoint['model_state_dict'])

    dict_mu = {label:[] for label in range(10)}

    t = tqdm(total=len(test_loader))
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            t.update(1)
            batch_x = batch_x.view((batch_x.shape[0], -1))
            mu, _ = vae.encode(batch_x)

            for i in range(BATCH_SIZE):
                label = batch_y[i].item()
                dict_mu[label].append(mu[i])

    l_points = []
    l_labels = []
    for label, mu in dict_mu.items():
        mu = np.array(torch.stack(mu))
        l_points.append(plt.scatter(mu[:,0], mu[:,1]))
        l_labels.append(label)

    plt.legend(l_points, l_labels)
    plt.show()


if __name__ == '__main__':
    # main()
    main_visualisation()

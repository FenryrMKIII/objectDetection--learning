import argparse
import sys
sys.path.append(r'C:\Users\DAA426\myWork\objectDetection-learning\classifier') # at some point, this should become import classifier when it will be made as a package and installed)
from classifier.architectures import classification 
import configparser
from PIL import Image 
import torch
import torchvision
from classifier.dataLoader.image import classificationDataSet
import pathlib
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train network on MNIST pyTorch dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-cfg', '--configuration', help='Neural Network Configuration File', required=True)
    parser.add_argument('-s', '--savePath', help='saving path for the weights', required=True)
    parser.add_argument('-b', '--batchSize', help='size of the batches', required=True)
    parser.add_argument('-e', '--epochs', help='number of epochs', required=True)
    args = parser.parse_args()

    # Parse arguments
    configFile = args.configuration
    savePath = Path(args.savePath)
    pathlib.Path(savePath).mkdir(parents=True, exist_ok=True)
    batchSize = int(args.batchSize)
    epochs = int(args.epochs)

    # Parse configuration file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(configFile)
    
    architecture = config['ARCHITECTURE']['architecture'] 
    optimizer = config['OPTIMIZER']['optimizer']
    loss = config['OPTIMIZER']['loss']
    if architecture == "LeNet5":
         arch = classification.LeNet5()
    if optimizer == "sgd":
        optimizer = optim.SGD(arch.parameters(), lr=0.001, momentum=0.9)
    if loss == "crossentropy":
        lossCompute = nn.CrossEntropyLoss()

    # Set computation device
    device = torch.device('cpu')

    # create dataset
    dataPath = Path(__file__).parents[3] / 'data' 
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), 
                                              torchvision.transforms.ToTensor()])
    trainingSet = torchvision.datasets.MNIST(root=dataPath, train=True,
                  download=False, transform=transform)
    testSet = torchvision.datasets.MNIST(root=dataPath, train=False,
                  download=False, transform=transform)

    # just take a look
    toPil = torchvision.transforms.ToPILImage()
    toPil(trainingSet[10][0]).show(title='sample {}'.format(trainingSet[10][1]))

    # create a loader to handle loading of data and potentially
    # multiprocessing
    trainLoader = DataLoader(trainingSet, batch_size=batchSize,
                            shuffle=True, num_workers=1)
    testLoader = DataLoader(testSet, batch_size=batchSize,
                            shuffle=False, num_workers=1)

    # prepare optimization loop
    for epoch in range(epochs):
        running_loss = 0.0
        print("starting epoch {:d}".format(epoch))
        arch.train()
        for i, data in enumerate(trainLoader):
            # split labels & inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = arch(inputs)
            loss = lossCompute(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % 2000 == 0) | (i==len(trainLoader)):    # print every 2000 mini-batches
                print('train statistics : after {} batches, average loss is {:.2f} [{:d}/{:d}]'.format(i,running_loss/2000,i*batchSize,len(trainingSet)))
                torch.save(arch.state_dict(), Path.joinpath(savePath, "mnist_LeNet5_" + str(i+1) + ".pth"))
                running_loss = 0
            
        arch.eval()
        nbCorrect = 0
        for i, data in enumerate(testLoader):
            inputs, labels = data
            outputs = arch(inputs)
            _, predicted_label = torch.max(outputs, 1)
            nbCorrect += torch.sum(predicted_label == labels).numpy()
        print("test statistics : accuracy equals {:.2f}".format(nbCorrect/len(testSet)))

    torch.save(arch.state_dict(), Path.joinpath(savePath, "mnist_LeNet5_final.pth"))
    print('Finished Training,')    

#!/bin/python
from optparse import OptionParser

import os
import shutil
import numpy as np

import torch
from torch.autograd import Variable

import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Custom imports
from Dataloader import *
import pretrainedmodels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(options):
    # Clear output directory
    if os.path.exists(options.outputDir):
        print ("Removing old directory!")
        shutil.rmtree(options.outputDir)
    os.mkdir(options.outputDir)

    # Create model
    if options.useTorchVisionModels:
        model = models.densenet161(pretrained=True)

        # Identify the name of the last layer
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)

        # Change the last layer
        inputDim = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputDim, options.numClasses)

        # TODO: Rectify the transform params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    else:
        # Use pretrained models library -  (pip install --upgrade pretrainedmodels)
        # https://github.com/Cadene/pretrained-models.pytorch
        model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1001, pretrained='imagenet+background')

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = model.mean
        std = model.std
        input_size = model.input_size[1]

        assert model.input_size[1] == model.input_size[2], "Error: Models expects different dimensions for height and width"
        assert model.input_space == "RGB", "Error: Data loaded in RGB format while the model expects BGR"

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    # Create dataloader
    dataTransform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    dataset = MyDataset(options.rootDir, split=Data.TRAIN, transform=dataTransform)
    dataLoader = DataLoader(dataset=dataset, num_workers=8, batch_size=options.batchSize, shuffle=True)
    assert options.numClasses == dataset.getNumClasses(), "Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!" % (dataset.getNumClasses(), options.numClasses)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(options.trainingEpochs):
        # Start training
        model.train()

        for iterationIdx, data in enumerate(dataLoader):
            X = data["data"]
            y = data["label"]

            # Move the data to PyTorch on the desired device
            X = Variable(X).float().to(device)
            y = Variable(y).long().to(device)

            # Get model predictions
            pred = model(X)

            # Optimize
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            # scheduler.step(val_loss)

            if iterationIdx % options.displayStep == 0:
                print("Epoch %d | Iteration: %d | Loss: %.5f" % (epoch, iterationIdx, loss))

        # Save model
        torch.save(model.state_dict(), os.path.join(options.outputDir, "model.pth"))

def test(options):
    # Clear output directory
    if not os.path.exists(options.outputDir):
        print("Error: Model directory does not exist!")
        exit(-1)

    # Create model
    if options.useTorchVisionModels:
        model = models.densenet161(pretrained=True)

        # Identify the name of the last layer
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)

        # Change the last layer
        inputDim = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputDim, options.numClasses)

        # TODO: Rectify the transform params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    else:
        # Use pretrained models library -  (pip install --upgrade pretrainedmodels)
        # https://github.com/Cadene/pretrained-models.pytorch
        model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1001, pretrained='imagenet+background')

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = model.mean
        std = model.std
        input_size = model.input_size[1]

        assert model.input_size[1] == model.input_size[2], "Error: Models expects different dimensions for height and width"
        assert model.input_space == "RGB", "Error: Data loaded in RGB format while the model expects BGR"

    # Move the model to desired device
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Create dataloader
    dataTransform = transforms.Compose([
        transforms.CenterCrop(input_size, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    dataset = MyDataset(options.rootDir, split=Data.TEST, transform=dataTransform)
    dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=options.batchSize, shuffle=False)

    # Save model
    modelCheckpoint = torch.load(os.path.join(outputDir, "model.pth"))
    model.load_state_dict(modelCheckpoint)
    print("Model restored!")

    gtLabels = []
    predictedLabels = []
    model.eval()
    for iterationIdx, data in enumerate(dataLoader):
        X = data["data"]
        y = data["label"]

        # Move the data to PyTorch on the desired device
        X = Variable(X).float().to(device)
        y = Variable(y).long().to(device)

        # Get model predictions
        outputs = model(X)

        # Check prediction
        _, preds = torch.max(outputs.data, dim=1)
        correctPred = torch.sum(preds == y.data)
        correctExamples = correctPred.item()

        # Add the labels
        gtLabels.append(data["label"])
        predictedLabels.append(preds.numpy())

        print("Iteration: %d | Correct examples: %d | Total examples: %d | Accuracy: %.5f" % (iterationIdx, correctExamples, len(predictedLabels[-1]), float(correctExamples) / len(predictedLabels[-1])))

    # Compute statistics
    gtLabels = np.array(gtLabels).flatten()
    predictedLabels = np.array(predictedLabels).flatten()

    print("GT labels shape:", gtLabels.shape)
    print("Predicted labels shape:", predictedLabels.shape)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    # Base options
    parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")
    parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
    parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
    parser.add_option("-o", "--outputDir", action="store", type="string", dest="outputDir", default="./output", help="Output directory")
    parser.add_option("-e", "--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Number of training epochs")
    parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=22, help="Batch Size (will be divided equally among different GPUs in multi-GPU settings)")
    parser.add_option("-d", "--displayStep", action="store", type="int", dest="displayStep", default=2, help="Display step where the loss should be displayed")

    # Input Reader Params
    parser.add_option("--rootDir", action="store", type="string", dest="rootDir", default="../data/", help="Root directory containing the data")
    parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=32, help="Number of classes in the dataset")

    parser.add_option("--useTorchVisionModels", action="store_true", dest="useTorchVisionModels", default=False, help="Use pre-trained models from the torchvision library")

    # Parse command line options
    (options, args) = parser.parse_args()
    print(options)

    if options.trainModel:
        print("Training model")
        train(options)

    if options.testModel:
        print("Testing model")
        test(options)

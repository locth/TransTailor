import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools
import pickle
import os
import gdown

##################################
# DEFINE CONSTANT
BATCH_SIZE = 128
NUM_WORKER = 20

FINETUNE_EPOCH = 15
FINETUNE_LR = 0.001
FINETUNE_MOMENTUM = 0.9

CHECKPOINT_URL = "https://drive.google.com/file/d/1VaQ2-ZVSt2kdMDHER34ghNCEnyJ-lNXu/view?usp=drive_link"
CHECKPOINT_DIR = "./checkpoint"
CHECKPOINT_NAME = "checkpoint_epoch_15.pt"

ALPHA_CHECKPOINT = ""
ALPHA_EPOCH = 10
ALPHA_LR = 0.001
ALPHA_MOMENTUM = 0.9

ROOT_DIR = "~/TransTailor"

##################################

def LoadData(numWorker, batchSize):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the CIFAR10 train_dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    kwargs = {'num_workers': numWorker, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batchSize, shuffle=True, **kwargs)

    # Load test_dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    kwargs = {'num_workers': 12, 'pin_memory': True} if device == 'cuda' else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batchSize, shuffle=False, **kwargs)

    return train_loader, test_loader

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
   
    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    return model

def ModelFinetune(device, model, train_loader, num_epochs, learningRate, momentum, checkpointEpoch):
    checkpoint_path = 'checkpoint/checkpoint_epoch_{epoch}.pt'

    print("\n===Fine-tune the pre-trained model to generate W_s*===")
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    epoch = checkpointEpoch

    for epoch in range(epoch, num_epochs):
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs))
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path.format(epoch=epoch + 1))
    
    return model

def ScalingFactorsInit(model, alphaCheckpoint):
    if os.path.exists(alphaCheckpoint):
        print("Load alpha from checkpoint!")
        with open(alphaCheckpoint, 'rb') as handle:
            scaling_factors = pickle.load(handle)
    else:
        print("Init alpha from scratch!")
        num_layers = len(model.features)
        scaling_factors = {}

        for i in range(num_layers):
            layer = model.features[i]
            if isinstance(layer, torch.nn.Conv2d):
                print(layer,layer.out_channels)
                scaling_factors[i] = torch.rand((1,layer.out_channels,1,1), requires_grad=True)
    return scaling_factors

def ScalingFactorsTraining(model, scaling_factors, num_epochs, learning_rate, momentum):
    for param in model.parameters():
        param.requires_grad = False
   
    criterion = torch.nn.CrossEntropyLoss()
    num_layers = len(model.features)

    print("\n===Train the factors alpha by optimizing the loss function===")

    params_to_optimize = itertools.chain(scaling_factors[sf] for sf in scaling_factors.keys())
    optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        print("Epoch " + str(epoch) + "/" + str(num_epochs))
        iter_count = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.shape[0]
            optimizer_alpha.zero_grad()
            outputs = inputs
            outputs.requires_grad = False

            for i in range(num_layers):
                if isinstance(model.features[i], torch.nn.Conv2d):
                    outputs = model.features[i](outputs)
                    outputs = outputs*scaling_factors[i].cuda()
                else:
                    outputs = model.features[i](outputs)

            outputs = torch.flatten(outputs, 1)
            classification_output = model.classifier(outputs)
            loss = criterion(classification_output, labels)
            loss.backward()
            optimizer_alpha.step()
    
    return scaling_factors

if __name__ == "__main__":
    print("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    batch_size = 64
    num_worker = 12

    print("LOAD DATASET: CIFAR10")
    train_loader, test_loader = LoadData(num_worker, batch_size)

    print("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)

    # LOAD MODEL FROM SAVE
    load_saved_model = input("Do you want to download checkpoint? [y/n]: ")
    if load_saved_model == "y": 
        if CHECKPOINT_URL:
            gdown.download(url=CHECKPOINT_URL, output='./checkpoint/', fuzzy=True)

    checkpoint_epoch = 0
    load_saved_model = input("Do you want to load model from checkpoint? [y/n]: ")
    if load_saved_model == "y":
        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        if os.path.isfile(checkpoint_path):
            # LOAD MODEL FROM SAVE'S PATH
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Load model successfully!")

            # GET CHECKPOINT NUMBER FROM SAVE'S NAME
            checkpoint_epoch = checkpoint['epoch']
            print("Load checkpoint: ", checkpoint_epoch)
        else:
            print("Cannot find saved model, use original pretrained model instead!")

    # CHECK NUMBER OF PARAMETERS
    print("Total params: ", sum(p.numel() for p in model.parameters()))
    print("Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # TARGET AWARE PRUNING: Fine tune the model on target dataset (CIFAR 10)
    model = ModelFinetune(device, model, train_loader, FINETUNE_EPOCH, FINETUNE_LR, FINETUNE_LR, checkpoint_epoch)
    
    scalingFactors = ScalingFactorsInit(model, ALPHA_CHECKPOINT)
    scalingFactors = ScalingFactorsTraining(model, scalingFactors, ALPHA_EPOCH, ALPHA_LR, ALPHA_MOMENTUM)
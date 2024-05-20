import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools
import pickle
import os

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

def model_finetune(device, model, train_loader, num_epochs, learningRate, momentum, checkpointEpoch):
    checkpoint_path = 'checkpoint/checkpoint_epoch_{epoch}.pt'

    print("\n===Fine-tune the pre-trained model to generate W_s*===")
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    epoch = checkpointEpoch

    for epoch in range(epoch, num_epochs):
        print("Epoch " + str(epoch) + "/" + str(num_epochs))
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path.format(epoch=epoch + 1))


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
    checkpoint_epoch = 0
    load_saved_model = input("Do you want to load model from save?   (y/n): ")
    if load_saved_model == "y":
        checkpoint_path = input("Enter the path to the saved model: ")
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

    # TARGET AWARE PRUNING
    num_epochs = 12
    learning_rate = 0.001
    momentum = 0.9

    model_finetune(device, model, train_loader, num_epochs, learning_rate, momentum, checkpoint_epoch)
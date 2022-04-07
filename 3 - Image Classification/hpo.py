#Import our dependencies.
#Below are some dependencies we need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

#Custom dependencies
import os 
import sys
import logging 
import argparse

# Avoid OS truncated error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Custom logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader):
    '''
    Yields test accuracy/loss of model from model and test data loader.
    Uses cross entropy for multi-class classification on batched test data. 
    '''
    logger.info(f"Testing model on test data")
    model.eval() 
    loss_acc = 0 
    correct_labels = 0 
    with torch.no_grad(): # Disable gradients and backprop for fast forward pass inference 
        for images, actual_labels in test_loader: # We don't need GPU device for inference 
            predicted_labels = model(images)
            loss = nn.CrossEntropyLoss(predicted_labels, actual_labels)
            predicted_label_indices = predicted_labels.argmax(dim=1, keepdim=True)
            loss_acc += loss.item() * images.size(0) # loss.item() gives loss of entire batch divided by size
            correct_labels += predicted_label_indices.eq(actual_labels.view_as(predicted_label_indices)).sum().item() 
            
        total_loss = loss_acc / len(test_loader.dataset)
        total_correct_labels = running_corrects/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, correct_labels, len(test_loader.dataset), 100.0 * total_correct_labels
        ))


def train(model, train_loader, criterion, optimizer):
    '''
    Trains model on loaded training data, by minimizing criterion using optimizer. 
    Includes debugging/profiling hooks to troubleshoot training process. 
    '''
    logger.info(f"Training model on complete training data")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    loss_acc = 0
    correct_labels = 0
    total_images = 0
    for images, actual_labels in train_loader:
        images = images.to(device)
        actual_labels = actual_labels.to(device)
        optimizer.zero_grad()
        predicted_labels = model(images)
        loss = criterion(predicted_labels, actual_labels)
        predicted_label_indices = predicted_labels.argmax(dim=1,  keepdim=True)
        loss_acc += loss.item() * images.size(0) # loss.item() gives entire batch loss divided by size 
        correct_labels += predicted_label_indices.eq(actual_labels.view_as(predicted_label_indices)).sum().item() 
        total_images += len(images) 
        loss.backward() # backpropagation
        optimizer.step() # step in direction of gradient minimizing loss 
        if total_images % 1000 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                total_images,
                len(train_loader.dataset),
                total_images / len(train_loader.dataset) * 100,
                loss.item(),
                correct_labels,
                total_images,
                correct_labels / total_images * 100
            ))
    total_loss = loss_acc / len(train_loader.dataset)
    total_acc = correct_labels/ len(train_loader.dataset)
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, correct_labels, len(train_loader.dataset), 100.0 * total_acc
    ))   
    return model
    
def net():
    ''' 
    Initialize pretrained model 
    '''
    model = models.resnet50(pretrained = True) # Pretrained convolutional model with 50 layers
    
    for parameter in model.parameters():
        parameter.requires_grad = False # Freeze pretrained model layers
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear( num_features, 256), # Add two fully connected layers
                             nn.ReLU(inplace = True), # Activation function 
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True) # Output should have 133 neurons, corresponding to 133 dog breed classes
                            )
    return model

def create_data_loaders(data, batch_size):
    '''
    Returns data loaders used to fetch training and test data 
    '''
    train_data_path = os.path.join(data, "train")
    test_data_path = os.path.join(data, "test")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader
    
def main(args):
    # Log info
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    logger.info(f"Hyperparameters: Learning rate: {args.lr},  Epsilon: {args.eps}, Weight Decay: {args.weight_decay}, Batch Size: {args.batch_size}, Epoch: {args.epochs}")
    logger.info(f"Data path: {args.data_dir}")
    logger.info(f"Model path: {args.model_dir}")
    logger.info(f"Output path: {args.output_dir}")
    
    # Initialize a model to GPU by calling the net function
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load data
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size)
        
    # Create your loss and optimizer, training fully connected output layer while freezing resnet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)
    
    # Train model on S3 trainign data  
    for epoch in range(args.epochs):
        # Train the model 
        logger.info(f"Epoch {epoch} - Starting training...")
        model=train(model, train_data_loader, criterion, optimizer)
         # Test the model to see its accuracy
        logger.info(f"Epoch {epoch} - Starting testing...")
        test(model, test_data_loader)
    
    # Save the trained model 
    logger.info("Saving model")
    torch.save(model, path)
    logger.info("Saved model")
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specifies all the hyperparameters we need to train your model.
    '''
    parser.add_argument(  "--batch_size", type=int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=3, metavar="E", help="number of epochs to train (default: 3)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 0.1)" )
    parser.add_argument( "--eps", type=float, default=1, metavar="EPS", help="eps (default: 1)" )
    parser.add_argument( "--weight_decay", type=float, default=0.01, metavar="WD", help="weight decay (default 0.01)" )
                        
    # Sagemaker Environment Variables locate training data, model location, and output folder in S3 bucket
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    
    args=parser.parse_args()
    
    main(args)

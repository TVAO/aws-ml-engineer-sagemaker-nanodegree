#Import our dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Custom libraries
import os
import sys
import logging
import argparse

# Fix OS truncation error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd

#Logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Hook and device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hook = smd.Hook.create_from_json_file()

def test(model, test_loader):
    '''
    Yields test accuracy and loss of model on test data. 
    Sets debugger hook to evaluation mode. 
    '''
    logger.info(f"Testing model on complete testing data")
    model.eval()
    hook.set_mode(smd.modes.EVAL) # set sm debugger hook 
    loss_acc = 0
    correct_labels = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, actual_labels in test_loader:
            images=images
            actual_labels=actual_labels
            predicted_labels=model(images)
            loss=criterion(predicted_labels, actual_labels)
            predicted_label_indices = predicted_labels.argmax(dim=1, keepdim=True)
            loss_acc += loss.item() * images.size(0) #calculate the running loss
            correct_labels += predicted_label_indices.eq(actual_labels.view_as(predicted_label_indices)).sum().item() 

        total_loss = loss_acc / len(test_loader.dataset)
        total_acc = correct_labels/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, correct_labels, len(test_loader.dataset), 100.0 * total_acc
        ))

def train(model, train_loader, criterion, optimizer):
    '''
    Returns model trained on data to minimize criterion using optimizer. 
    Sets debugger hook to training mode. 
    '''
    logger.info(f"Training model on complete training data")
    model.train()
    hook.set_mode(smd.modes.TRAIN) # set debugger hook mode to TRAIN
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
        loss_acc += loss.item() * images.size(0) # loss.item() gives total batch loss divided by size 
        correct_labels += predicted_label_indices.eq(actual_labels.view_as(predicted_label_indices)).sum().item() 
        total_images += len(images) #keep count of running samples
        loss.backward()
        optimizer.step()
        if total_images % 1000 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                total_images,
                len(train_loader.dataset),
                total_images / len(train_loader.dataset) * 100,
                loss.item(),
                correct_labels,
                total_images,
                correct_labels/ total_images * 100
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
    model = models.resnet50(pretrained = True) # Pretrained resnet50 convolutional model with 50 layers
    
    for parameter in model.parameters():
        parameter.requires_grad = False #Freeze pretrained model layers 
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear( num_features, 256), 
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True) # Output should be 133 neurons, corresponding to 133 dog breed labels
                            )
    return model

def create_data_loaders(data, batch_size):
    '''
    Create data loader for data.
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
    # Logging
    logger.info(f"Running device {device}")
    logger.info(f"Hyperparameters : Learning rate: {args.lr},  Epsilon: {args.eps}, Weight decay: {args.weight_decay}, Batch size: {args.batch_size}, Epoch: {args.epochs}")
    logger.info(f"Data path: {args.data_dir}")
    logger.info(f"Model path: {args.model_dir}")
    logger.info(f"Output path: {args.output_dir}")
    
    # Initialize a model by calling the net function
    model=net()
    model = model.to(device)
    
    # Register model hook to capture tensors from training model 
    hook.register_module(model)
    
    # Load data 
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size )
    
    # Create your loss and optimizer
    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion) # register loss 
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)
    
    for epoch_no in range(1, args.epochs +1 ):
        #Call the train function to start training your model from training data in S3 
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model=train(model, train_data_loader, criterion, optimizer)
        # Test the model to see its accuracy
        logger.info(f"Epoch {epoch_no} - Starting Testing phase.")
        test(model, test_data_loader)

    # Save the trained model
    logger.info("Saving model...")
    torch.save(model, path)
    logger.info("Saved model...")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specifies any training args that we may need
    '''
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=3, metavar="E", help="number of epochs to train (default: 3)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
    parser.add_argument( "--eps", type=float, default=1, metavar="EPS", help="eps (default: 1)" )
    parser.add_argument( "--weight_decay", type=float, default=0.01, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 0.01)" )
                        
    # Sagemaker environment variables to locate training data, model dir and output in S3 bucket
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)

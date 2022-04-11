# PyTorch libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Custom libraries 
import json
import logging
import sys
import os
from PIL import Image
import io
import requests

# Custom logger 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Image type support 
JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ] 

# Enable GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def net():
    model = models.resnet50(pretrained = True)
    for parameter in model.parameters():
        parameter.requires_grad = False 
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear( num_features, 256), nn.ReLU(inplace = True), nn.Linear(256, 133), nn.ReLU(inplace = True))
    return model


#Override how model is loaded 
def model_fn(model_folder):
    logger.info("Inside model_fn function")
    logger.info(f"Read model from folder: {model_folder}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run on device: {device}")
    model = net().to(device)
    with open(os.path.join(model_folder, "model.pth"), "rb") as f:
        logger.info("Start loading model...")
        model.load_state_dict(torch.load(f, map_location = device))
        logger.info("Successfully loaded model")
    return model

# Override the default input_fn
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing input image.')
    # Process an image uploaded to the endpoint
    logger.info(f'Request Content-Type is: {content_type}')
    logger.info(f'Request Body is: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        logger.info(f"Returning image of type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested unsupported Content-Type: {content_type}, Accepted Content-Type: {ACCEPTED_CONTENT_TYPE}")

# Override the default predict_fn
def predict_fn(image, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Start image classification...")
    transform =  transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor() ])
    logger.info("Transform image")
    image = transform(image)
    if torch.cuda.is_available():
        image = image.cuda() # Put image data to GPU 
    logger.info("Completed transforming image")
    model.eval()
    with torch.no_grad():
        logger.info("Start model invocation")
        prediction = model(image.unsqueeze(0))
    return prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import random
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import  Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import torchvision.transforms as transforms
import albumentations as A
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import wandb
wandb.init(mode= 'disabled')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

preprocess = A.Compose([
                A.Rotate(limit=2, p=0.5),
                A.GaussNoise(var_limit=(5.0, 10.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        ])

class CustomDataset(Dataset):
    def __init__(self,images,gt_text,processor,transform):
        self.images=images
        self.texts=gt_text
        self.processor=processor
        self.transform=transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name=self.images[idx]
        image=Image.open(image_name)
        text=self.texts[idx]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = np.array(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1) 
            image = np.repeat(image, 3, axis=-1)   

        image = (image * 255).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        image = Image.fromarray(image)

        # Resize image
        image = image.resize((256,64), Image.BILINEAR)

        # Convert image back to numpy array
        image = np.array(image) / 255.0

        # Ensure the image has shape (3, height, width)
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            print(image.shape)
            
        pixel_values = self.processor(image,return_tensors="pt").pixel_values
        pixel_values=pixel_values.squeeze() #extra dim htao
        
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids
        labels= labels[:, :512]
        labels=labels.squeeze()
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_vals = [it['pixel_values'] for it in batch]
    pixel_values=torch.stack(pixel_vals)
    
    labels=[it['labels'] for it in batch]
    labels=pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {'pixel_values':pixel_values, 'labels':labels}


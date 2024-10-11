import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import  Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import torchvision.transforms as transforms
import albumentations as A
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import tqdm as tqdm
import argparse
import wandb
wandb.init(mode= 'disabled')

if __name__=='__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42, type=int, help='Seeding Number')
    parser.add_argument('--lr', default=3e-5, type=float)

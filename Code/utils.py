import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import torch # type: ignore
import re
from PIL import Image # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split # type: ignore
import random
from transformers import TrOCRProcessor, VisionEncoderDecoderModel # type: ignore
from transformers import  Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback # type: ignore
from torch.nn.utils.rnn import pad_sequence # type: ignore
from evaluate import load # type: ignore
import torchvision.transforms as transforms # type: ignore
import albumentations as A # type: ignore
from torch.optim import AdamW # type: ignore
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts# type: ignore
import torch.nn.functional as F # type: ignore
import wandb  # type: ignore
wandb.init(mode= 'disabled')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


            

class CustomDataset(Dataset):
    def __init__(self,images,gt_text,processor,preprocess=True):
        self.images=images
        self.texts=gt_text
        self.processor=processor
        self.preprocess=preprocess

        if self.preprocess:
            self.transform = A.Compose([
                A.OneOf([
                    A.Rotate(limit=2, p=1.0),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.ElasticTransform(alpha=0.3, sigma=50.0, alpha_affine=None, p=1.0),
                    A.OpticalDistortion(distort_limit=0.03, shift_limit=0.03, p=1.0),
                    A.CLAHE(clip_limit=2, tile_grid_size=(4, 4), p=1.0),
                    A.Affine(scale=(0.95, 1.05), translate_percent=(0.01, 0.01), shear=(-2, 2), p=1.0),
                    A.Perspective(scale=(0.01, 0.03), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GridDistortion(num_steps=3, distort_limit=0.02, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0)
                ], p=0.8),
            ])
        else:
            self.transform = A.Compose([])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name=self.images[idx]
        image=Image.open(image_name)
        text=self.texts[idx]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        

        if not self.preprocess:
            image = image.convert('L')  # Convert to grayscale
            image = np.array(image)
            image = np.stack([image] * 3, axis=-1)  # Repeat to create 3 channels
        else:
            image = np.array(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1) 
            image = np.repeat(image, 3, axis=-1)   

        image = (image * 255).astype(np.uint8)
        if self.preprocess:
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

def beam_search_loss(logits, labels, beam_size=6, length_penalty=3.0, alpha=0.6, label_smoothing=0.1):
    batch_size, seq_len, vocab_size = logits.shape
    beam_scores, beam_indices = logits.topk(beam_size, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    beam_log_probs = log_probs.gather(-1, beam_indices)

    mask = (labels != -100).float()
    sequence_lengths = mask.sum(dim=1)

    labels = labels.long()
    valid_labels = labels.clamp(min=0, max=vocab_size - 1)

    smooth_labels = torch.zeros_like(logits).scatter_(-1, valid_labels.unsqueeze(-1), 1)
    smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / vocab_size

    beam_loss = -(smooth_labels * log_probs).sum(dim=-1)
    beam_loss = beam_loss.masked_fill(mask == 0, 0).sum(dim=1)

    normalized_beam_loss = beam_loss / (sequence_lengths ** length_penalty)
    log_likelihood = -normalized_beam_loss
    normalized_log_likelihood = log_likelihood / sequence_lengths.pow(alpha)

    losses = [-normalized_log_likelihood]
    stacked_losses = torch.stack(losses, dim=1)
    min_loss = stacked_losses.min(dim=1)[0]

    return min_loss.mean()

class BeamSearchTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pixel_values = inputs.pop("pixel_values")

        outputs = model(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits

        loss = beam_search_loss(logits, labels, label_smoothing=self.label_smoothing)

        return (loss, outputs) if return_outputs else loss

def focal_beam_search_loss(logits, labels, beam_size=10, length_penalty=2.0, alpha=0.6, label_smoothing=0.1, gamma=2.0):
    batch_size, seq_len, vocab_size = logits.shape
    beam_scores, beam_indices = logits.topk(beam_size, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    beam_log_probs = log_probs.gather(-1, beam_indices)

    mask = (labels != -100).float()
    sequence_lengths = mask.sum(dim=1)

    labels = labels.long()
    valid_labels = labels.clamp(min=0, max=vocab_size - 1)

    smooth_labels = torch.zeros_like(logits).scatter_(-1, valid_labels.unsqueeze(-1), 1)
    smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / vocab_size

    # Focal Loss integration
    probs = F.softmax(logits, dim=-1)
    focal_weight = (1 - probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)) ** gamma
    beam_loss = -(smooth_labels * log_probs).sum(dim=-1) * focal_weight
    beam_loss = beam_loss.masked_fill(mask == 0, 0).sum(dim=1)

    normalized_beam_loss = beam_loss / (sequence_lengths ** length_penalty + 1e-6)  # Add small epsilon
    log_likelihood = -normalized_beam_loss
    normalized_log_likelihood = log_likelihood / (sequence_lengths.pow(alpha) + 1e-6)  # Add small epsilon

    losses = [-normalized_log_likelihood]
    stacked_losses = torch.stack(losses, dim=1)
    min_loss = stacked_losses.min(dim=1)[0]

    return min_loss.mean()

class FocalBeamSearchTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pixel_values = inputs.pop("pixel_values")

        outputs = model(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits

        loss = focal_beam_search_loss(logits, labels, label_smoothing=self.label_smoothing, gamma=self.gamma)

        return (loss, outputs) if return_outputs else loss

def focal_beam_search_loss2(logits, labels, beam_size=10, length_penalty=1.1, alpha=0.6, label_smoothing=0.1, gamma=2.0, temperature=1.0):
    scaled_logits = logits / temperature
    batch_size, seq_len, vocab_size = scaled_logits.shape
    beam_scores, beam_indices = scaled_logits.topk(beam_size, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    beam_log_probs = log_probs.gather(-1, beam_indices)

    mask = (labels != -100).float()
    sequence_lengths = mask.sum(dim=1)

    labels = labels.long()
    valid_labels = labels.clamp(min=0, max=vocab_size - 1)

    smooth_labels = torch.zeros_like(scaled_logits).scatter_(-1, valid_labels.unsqueeze(-1), 1)
    smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / vocab_size

    probs = F.softmax(scaled_logits, dim=-1)
    focal_weight = (1 - probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)) ** gamma
    beam_loss = -(smooth_labels * log_probs).sum(dim=-1) * focal_weight
    beam_loss = beam_loss.masked_fill(mask == 0, 0).sum(dim=1)

    normalized_beam_loss = beam_loss / (sequence_lengths ** length_penalty + 1e-6)
    log_likelihood = -normalized_beam_loss
    normalized_log_likelihood = log_likelihood / (sequence_lengths.pow(alpha) + 1e-6)

    losses = [-normalized_log_likelihood]
    stacked_losses = torch.stack(losses, dim=1)
    min_loss = stacked_losses.min(dim=1)[0]

    return min_loss.mean()

class FocalBeamSearchTrainer2(Trainer):
    def __init__(self, *args, label_smoothing=0.1, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pixel_values = inputs.pop("pixel_values")

        outputs = model(pixel_values=pixel_values, labels=labels)

        # Handle the case where outputs is a tensor (logits) or an object with logits attribute
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        temperature = model.temperature.item() if hasattr(model, 'temperature') else 1.0

        loss = focal_beam_search_loss2(logits, labels, label_smoothing=self.label_smoothing, gamma=self.gamma, temperature=temperature)

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Create a scheduler using CosineAnnealingWarmRestarts
        """
        if optimizer is None:
            optimizer = self.optimizer
        return CosineAnnealingWarmRestarts(optimizer, T_0=num_training_steps // 15, T_mult=2, eta_min=1e-6)


class CosineAnnealingWarmRestartsSchedulerCallback(TrainerCallback):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)

    def on_step_end(self, args, state, control, **kwargs):
        self.scheduler.step()

class TemperatureScaledTrOCR(torch.nn.Module):
    def __init__(self, model, temperature=1.0):
        super(TemperatureScaledTrOCR, self).__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature)

    def forward(self, pixel_values, labels=None):
        if labels is not None:
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            outputs.logits = outputs.logits.float() / self.temperature
            return outputs
        else:
            logits = self.model(pixel_values=pixel_values).logits
            return logits.float() / self.temperature

    def generate(self, pixel_values, **kwargs):
        return self.model.generate(pixel_values, **kwargs)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the base model
        self.model.save_pretrained(save_directory)

        # Save the temperature parameter
        temperature_path = os.path.join(save_directory, "temperature.pt")
        torch.save(self.temperature, temperature_path)

    @classmethod
    def from_pretrained(cls, load_directory):
        base_model = VisionEncoderDecoderModel.from_pretrained(load_directory)
        temperature_path = os.path.join(load_directory, "temperature.pt")
        temperature = torch.load(temperature_path)

        model = cls(base_model, temperature.item())
        return model

class SLiCTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")

        # Initializing lambda
        lambda_reg = 0.01

        outputs = model(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        logits = logits.float()

        # Ensure labels are within the valid range
        vocab_size = logits.size(-1)
        labels = torch.clamp(labels, min=0, max=vocab_size - 1)

        temperature = model.temperature.item() if hasattr(model, 'temperature') else 1.0

        # Compute similarity scores
        similarity_scores = self.calculate_similarity(logits, labels)

        # Apply SLIC Loss
        slic_loss = self.compute_calibration_loss(logits, labels, similarity_scores)

        # Apply KL Divergence Loss
        kl_loss = self.compute_kl_divergence(logits, labels)

        # Apply Focal Loss
        focal_loss = self.compute_focal_loss(logits, labels)

        # Total Loss
        total_loss = slic_loss + lambda_reg * (kl_loss + focal_loss)

        # clearing unnecessary tensors
        del similarity_scores, slic_loss, kl_loss, focal_loss
        torch.cuda.empty_cache()

        return (total_loss, outputs) if return_outputs else total_loss

    def compute_calibration_loss(self, logits, labels, similarity_scores):
        batch_size, seq_len, vocab_size = logits.size()
        device = logits.device

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Generate positive and negative samples
        pos_samples = labels.unsqueeze(-1)
        neg_samples = torch.randint(0, vocab_size, (batch_size, seq_len, 1), device=device)

        # Compute losses
        l_rank = self.compute_rank_loss(log_probs, pos_samples, neg_samples)
        l_margin = self.compute_margin_loss(log_probs, similarity_scores, pos_samples, neg_samples)

        # Combine losses (you may want to add weights to different loss components)
        calibration_loss = l_rank + l_margin

        del log_probs, pos_samples, neg_samples, l_rank, l_margin
        torch.cuda.empty_cache()

        return calibration_loss

    def compute_rank_loss(self, log_probs, pos_samples, neg_samples):
        beta = 0.1
        l_rank = torch.max(torch.zeros_like(log_probs[:, :, 0]),
                           beta - log_probs.gather(-1, pos_samples).squeeze(-1) +
                           log_probs.gather(-1, neg_samples).squeeze(-1)).mean()
        # print("Rank Loss : ",l_rank)
        return l_rank

    def compute_margin_loss(self, log_probs, similarity_scores, pos_samples, neg_samples):
        beta = 0.1
        l_margin = torch.max(torch.zeros_like(log_probs[:, :, 0]),
                             beta * (similarity_scores.gather(-1, pos_samples).squeeze(-1) -
                                     similarity_scores.gather(-1, neg_samples).squeeze(-1)) -
                             log_probs.gather(-1, pos_samples).squeeze(-1) +
                             log_probs.gather(-1, neg_samples).squeeze(-1)).mean()

        # print("Margin Loss : ",l_margin)
        return l_margin


    def compute_kl_divergence(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        target_probs = F.one_hot(labels, num_classes=logits.size(-1)).float()
        kl_div_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

        # print("KL Divergence Loss : ",kl_div_loss)
        return kl_div_loss

    def compute_focal_loss(self, logits, labels, gamma=2.0):
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        # print("Focal Loss : ",focal_loss)
        return focal_loss

    def calculate_similarity(self, logits, labels):
        batch_size, seq_len, vocab_size = logits.size()
        device = logits.device

        # Get token embeddings
        token_embeddings = self.model.get_output_embeddings().weight  # Shape: [vocab_size, embedding_dim]

        # Compute logits_softmax once
        logits_softmax = F.softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

        ru_ee_first_col = torch.zeros(vocab_size, 1, device=device)
        ru_ee_first_col[0] = 1.0  # Assuming 0 is the padding token index

        # Part 1: Fu(e, e) contribution
        similarity_scores = torch.matmul(logits_softmax, token_embeddings)  # [batch_size, seq_len, embedding_dim]
        similarity_scores = torch.matmul(similarity_scores, token_embeddings.t())  # [batch_size, seq_len, vocab_size]

        # Part 2: Ru(e, e) contribution
        ru_contribution = torch.matmul(logits_softmax, ru_ee_first_col)  # [batch_size, seq_len, 1]

        # Combine both parts
        similarity_scores = (similarity_scores + ru_contribution) / 2

        return similarity_scores


    def beam_search_decode(self, model, pixel_values, beam_size=10, max_length=128):
        # Implement beam search decoding
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        encoder_outputs = model.encoder(pixel_values=pixel_values)
        input_ids = torch.full((batch_size * beam_size, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        for step in range(max_length):
            outputs = model.decoder(input_ids=input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state.repeat_interleave(beam_size, dim=0))
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            vocab_size = next_token_scores.size(-1)

            next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            next_tokens = next_tokens % vocab_size
            next_beam_scores = torch.gather(next_token_scores, -1, next_tokens.unsqueeze(-1)).squeeze(-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            beam_scores = next_beam_scores.view(-1)

            if (next_tokens == model.config.eos_token_id).any():
                break

        return input_ids.view(batch_size, beam_size, -1)[:, 0, :]

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if optimizer is None:
            optimizer = self.optimizer
        return CosineAnnealingWarmRestarts(optimizer, T_0=num_training_steps // 10, T_mult=2, eta_min=1e-6)

class TemperatureScalingModel(torch.nn.Module):
    def __init__(self, model, temperature=0.999):
        super(TemperatureScalingModel, self).__init__()
        self.model = model
        self.config = model.config
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature)

    def forward(self, pixel_values, labels=None):
        if labels is not None:
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            outputs.logits = outputs.logits.float() / self.temperature
            return outputs
        else:
            logits = self.model(pixel_values=pixel_values).logits
            return logits.float() / self.temperature

    def generate(self, pixel_values, **kwargs):
        return self.model.generate(pixel_values, **kwargs)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    
        self.model.save_pretrained(save_directory)
        temperature_path = os.path.join(save_directory, "temperature.pt")
        torch.save(self.temperature, temperature_path)

    @classmethod
    def from_pretrained(cls, load_directory):
        base_model = VisionEncoderDecoderModel.from_pretrained(load_directory)
        temperature_path = os.path.join(load_directory, "temperature.pt")
        temperature = torch.load(temperature_path)

        model = cls(base_model, temperature.item())
        return model

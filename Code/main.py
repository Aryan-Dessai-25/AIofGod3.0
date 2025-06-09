import numpy as np # type: ignore
import pandas as pd # type: ignore # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import torch # type: ignore
import re
from PIL import Image # type: ignore 
from torch.utils.data import Dataset, DataLoader, random_split # type: ignore

from transformers import TrOCRProcessor, VisionEncoderDecoderModel # type: ignore
from transformers import  Trainer, TrainingArguments, EarlyStoppingCallback,  GenerationConfig, get_cosine_schedule_with_warmup# type: ignore
from torch.nn.utils.rnn import pad_sequence # type: ignore
from evaluate import load # type: ignore
import torchvision.transforms as transforms # type: ignore
import albumentations as A # type: ignore
from torch.optim import AdamW # type: ignore
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau # type: ignore
import tqdm as tqdm # type: ignore
import argparse
from utils import CustomDataset, collate_fn, seed_everything, BeamSearchTrainer, beam_search_loss, FocalBeamSearchTrainer, focal_beam_search_loss
from utils import FocalBeamSearchTrainer2, focal_beam_search_loss2, CosineAnnealingWarmRestartsSchedulerCallback, TemperatureScaledTrOCR, SLiCTrainer, TemperatureScalingModel
import torch.nn.functional as F # type: ignore
import wandb # type: ignore
wandb.init(mode= 'disabled')
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

if __name__=='__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42, type=int, help='Seeding Number')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--wt_decay', default=0.01, type=float)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--eval_steps', default=120, type=int)
    parser.add_argument('--save_steps', default=1200, type=int)
    parser.add_argument('--grad_acc', default=8, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    args = parser.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    generation_config = GenerationConfig(
        max_length=128,
        no_repeat_ngram_size=3
    )
    wer = load("wer")
    cer = load("cer")
    def compute_metrics(eval_pred):
        processor=TTrOCRProcessor.from_pretrained("qantev/trocr-large-spanish", do_rescale=False)
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(-1)

        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = []
        for label in labels:
            label_filtered = [token for token in label if token != -100]
            decoded_label = processor.tokenizer.decode(label_filtered, skip_special_tokens=True)
            decoded_labels.append(decoded_label)

        wer_score = wer.compute(predictions=decoded_preds, references=decoded_labels)
        cer_score = cer.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer_score, "cer": cer_score}


    train_textdf=pd.read_csv('./aiog3/Public_data/train.csv')
    train_image_dir='./aiog3/Public_data/train_images'
    train_images=[f'{train_image_dir}/{f}' for f in sorted(os.listdir(train_image_dir))]

    train_gt=train_textdf['transcription'].values.tolist()

    processor = TrOCRProcessor.from_pretrained("./Results/badam7_beam", do_rescale=False)
    base_model = VisionEncoderDecoderModel.from_pretrained("./Results/badam7_beam")
    model = TemperatureScalingModel(base_model)


    model.model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.model.config.pad_token_id = processor.tokenizer.pad_token_id

    model=model.to(device)
    custom_ds=CustomDataset(train_images,train_gt,processor, preprocess=False)
    val_size = int(len(custom_ds)*args.val_ratio)
    train_size=len(custom_ds)-val_size
    custom_train, custom_val = random_split(custom_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

   
    training_args = TrainingArguments(
        output_dir="./Results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=args.wt_decay,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        fp16=True,
        report_to=None,
        gradient_accumulation_steps=args.grad_acc,
        dataloader_num_workers=4,
        save_steps=args.save_steps,
        dataloader_pin_memory=False,
        greater_is_better=False,
        warmup_steps=1000,
        warmup_ratio=0.1
        
    )
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay,eps=1e-8)
    
   
    num_training_steps = len(custom_train) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=custom_train,
    #     eval_dataset=custom_val,
    #     data_collator=collate_fn,
    #     optimizers=(optimizer, scheduler),
    #     #optimizers=optimizer,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    #     compute_metrics=compute_metrics,
        
    # )
    # trainer = BeamSearchTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=custom_train,
    #     eval_dataset=custom_val,
    #     data_collator=collate_fn,
    #     optimizers=(optimizer, scheduler),
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    #     compute_metrics=compute_metrics,
    #     label_smoothing=0.1
    # )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1000, num_training_steps=num_training_steps)


    trainer = SLiCTrainer(
        model=model,
        args=training_args,
        train_dataset=custom_train,
        eval_dataset=custom_val,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        label_smoothing=0.1,
        gamma=2.0,
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    trainer.train()

    model.save_pretrained("./Results/bskunduwer")
    processor.save_pretrained("./Results/bskunduwer")
    learned_temperature=model.temperature.item()
    torch.save({"temperature": learned_temperature}, "./Results/bskunduwer/learned_temperature.pth")

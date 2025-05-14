#!/usr/bin/env python
# coding: utf-8

import os
import gc
import torch
import time
import json
import logging
from tqdm import tqdm
from datasets import DatasetDict, load_dataset, load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    WhisperFeatureExtractor, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    SchedulerType, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import evaluate
import argparse
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Whisper model on Swiss German")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="The name of the dataset to use for training (via the datasets library).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each saving",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between each evaluation",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--keep_n_percent", type=float, default=1, help="Percentage of the dataset to keep for the training for memory purposes.")
    parser.add_argument("--language", type=str, default="german", help="Language used for the finetuning.")
    parser.add_argument("--data_augmentation", action="store_true", help="If True, apply data augmentation to the data.")
    
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    data_augmentation: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        audio_array = [sample['audio']['array'] for sample in features]
        sentences = [sample['sentence'] for sample in features]

        if self.data_augmentation:
            rates = np.random.choice([0.9, 1.0, 1.1], len(audio_array), p=[0.2, 0.6, 0.2])
            audio_array = [librosa.effects.time_stretch(audio_array[i], rate=rates[i]) for i in range(len(audio_array))]
 
        batch_features = self.processor.feature_extractor(audio_array, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt")
        batch = self.processor.feature_extractor.pad(batch_features, return_tensors="pt")

        # get the tokenized label sequences and pad them to max length
        label_features = self.processor.tokenizer(sentences)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def compute_metrics_gen(pred, metric, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def main():

    args = parse_args()
    hyperparams = vars(args)
    hyperparams_file = os.path.join(args.output_dir, 'config_params.json')
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Initialize WER metric
    metric = evaluate.load("wer")

    # Load Whisper model & processor
    model_name = args.model_name_or_path                                               
    processor = WhisperProcessor.from_pretrained(model_name, language=args.language, task="transcribe")                                                                                                                                                        
                                                                                                                                                    
    # Loading audio data                                                                                                                                                                                                                                                                   
    audio_dataset = load_from_disk(args.train_dataset)
    audio_dataset['train'] = audio_dataset['train'].cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
    audio_dataset['test'] = audio_dataset['test'].cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

    # Function to keep only n% of each split                                                                                                        
    def keep_n_percent(dataset_dict, n):                                                                                                              
        return DatasetDict({                                                                                                                          
            split: dataset.select(range(int(len(dataset) * n)))  # Keep only n%                                                                      
            for split, dataset in dataset_dict.items()                                                                                           
        })

    audio_dataset = keep_n_percent(audio_dataset, args.keep_n_percent)
    print(audio_dataset, flush=True)                                                                                                                              

    # Load model from checkpoint                                                                                                                      
    model = WhisperForConditionalGeneration.from_pretrained(model_name)                                                                                                                                         
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    # Legacy way of setting the language and task arguments
    model.generation_config.forced_decoder_ids = None 

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                            
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(                                                                                        
        processor=processor,                                                                                                                          
        decoder_start_token_id=model.config.decoder_start_token_id,
        data_augmentation=args.data_augmentation                                                                             
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.num_warmup_steps,
        max_steps=args.max_train_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.saving_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False
    )                                                                                                                                               

    compute_metrics = partial(compute_metrics_gen, metric=metric, processor=processor)
    
    # Initialize Trainer                                                                                                                              
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=audio_dataset["train"],
        eval_dataset=audio_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        print(f"Found checkpoint at {last_checkpoint}. Resuming training from there.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Start training; if last_checkpoint is None, training will start from scratch.
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()
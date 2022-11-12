import nltk

import transformers
import datasets
import re

import argparse

import json

import utils

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--training_args', type=str, required=True, help='Path to the training args (json)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate the model on the dev set')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')
    parser.add_argument("--train", type=str, required=True, help="Train file(s) separated by commas")
    parser.add_argument("--dev", type=str, required=True, help="Dev file(s) separated by commas")
    parser.add_argument("--test", type=str, required=True, help="Test file(s) separated by commas")

    args = parser.parse_args()
    return args

def train(args,model,tokenizer,train_dataset,eval_dataset):
    
    train_args_dict = json.load(open(args.training_args))
    train_args_dict.update({
        "output_dir" : args.output_dir,
        "logging_dir" : args.output_dir,
        "do_train" : args.do_train,
        "do_eval" : args.do_eval,
        "do_predict" : args.do_predict
    })

    data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

    train_args = transformers.TrainingArguments(**train_args_dict)

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=utils.compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

def predict(args,model,tokenizer,test_dataset):
    pass

def main():
    args = init_args()
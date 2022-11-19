import nltk

import torch
import transformers
import datasets
import os

import argparse

from tqdm import tqdm
import json

import utils

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', type=int, default=42, help='Random state')

    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum length of the input sequence')
    parser.add_argument('--training_args', type=str, required=True, help='Path to the training args (json)')
    parser.add_argument('--n_gpu', type=str, default="0", help='GPU(s) index to use')

    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate the model on the dev set')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')

    parser.add_argument('--pred_batch_size', type=int, default=8, help='Batch size for predictions')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data')
    parser.add_argument("--train", type=str, required=True, help="Train file(s) separated by commas")
    parser.add_argument("--dev", type=str, required=True, help="Dev file(s) separated by commas")
    parser.add_argument("--test", type=str, required=True, help="Test file(s) separated by commas")
    
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output')
    

    args = parser.parse_args()
    return args

def train(args,model,tokenizer,train_dataset,eval_dataset):
    encoding_args = {
        "max_length" : args.max_len,
        "padding" : "max_length",
        "truncation" : True,
        "return_tensors" : "pt",
        "is_split_into_words" : True
    }

    decoding_args = {
        "skip_special_tokens" : True
    }

    transformers.set_seed(args.random_state)
    
    train_args_dict = json.load(open(args.training_args))
    train_args_dict.update({
        "output_dir" : args.output_dir,
        "logging_dir" : args.output_dir,
        "do_train" : args.do_train,
        "do_eval" : args.do_eval,
        "do_predict" : args.do_predict
    })

    tokenized_dataset_train = train_dataset.map(lambda x : utils.tokenize_and_align_labels(x,tokenizer,encoding_args),batched=True)
    if args.do_eval:
        tokenized_dataset_eval = eval_dataset.map(lambda x : utils.tokenize_and_align_labels(x,tokenizer,encoding_args),batched=True)

    data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

    train_args = transformers.TrainingArguments(**train_args_dict)

    trainer_args = {
        "model" : model,
        "args" : train_args,
        "train_dataset" : tokenized_dataset_train,
        "tokenizer" : tokenizer,
        "compute_metrics" : lambda x :utils.compute_metrics(x,model.config.id2label),
        "data_collator" : data_collator
    }

    if args.do_eval:
        trainer_args.update({
            "eval_dataset" : tokenized_dataset_eval
        })

    trainer = transformers.Trainer(**trainer_args)

    trainer.train()

    # Save the model
    model.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)

def predict(args,model,tokenizer,test_dataset):
    encoding_args = {
        "max_length" : args.max_len,
        "padding" : "max_length",
        "truncation" : True,
        "return_tensors" : "pt",
        "is_split_into_words" : True
    }

    decoding_args = {
        "skip_special_tokens" : True
    }

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tokenized_dataset = test_dataset.map(lambda x : utils.tokenize_and_align_labels(x,tokenizer,encoding_args),batched=True,remove_columns=test_dataset.column_names)

    data_loader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=args.pred_batch_size,
        shuffle=False
    )

    with torch.no_grad():
        predictions = []
        for batch in tqdm(data_loader,desc="Predicting"):
            model.to(device)
            for k in batch.keys():
                for i_tensor in range(len(batch[k])):
                    batch[k][i_tensor] = batch[k][i_tensor].numpy()
                batch[k] = torch.tensor(batch[k]).to(device)
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits,dim=2).cpu().numpy().tolist())
    
    # predictions = [model.config.id2label[p] for p in predictions]
    for i_pred in range(len(predictions)):
        for j_pred in range(len(predictions[i_pred])):
            predictions[i_pred][j_pred] = model.config.id2label[predictions[i_pred][j_pred]]

    # save predictions
    with open(os.path.join(args.output_dir,"predictions.txt"),"w") as f:
        # f.write("\n".join(predictions))
        for data, prediction in zip(test_dataset,predictions):
            tokens = data["tokens"]
            bio_tags = data["bio_tags"]

            for token, bio_tag, pred in zip(tokens,bio_tags,prediction):
                f.write(f"{token}\t{model.config.id2label[bio_tag]}\t{pred}\n")
            f.write("\n")

def main():
    args = init_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu

    with open(os.path.join(args.data_dir,"labels.txt"),"r") as f:
        labels = f.read().splitlines()
        labels = {l:i for i,l in enumerate(labels)}
    
    label2id = labels
    id2label = {v:k for k,v in labels.items()}
    num_labels = len(labels)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = transformers.AutoModelForTokenClassification.from_pretrained(args.model_name_or_path,id2label=id2label,label2id=label2id,num_labels=num_labels)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.do_train:
        print("Loading train dataset")
        train_paths = [os.path.join(args.data_dir,p) for p in args.train.split(",")]
        print("Loading dev dataset")
        eval_paths = [os.path.join(args.data_dir,p) for p in args.dev.split(",")]
        train_dataset = utils.open_dataset(train_paths,labels)
        eval_dataset = utils.open_dataset(eval_paths,labels)
        print("Start training...")
        train(args,model,tokenizer,train_dataset,eval_dataset)
    if args.do_predict:
        test_paths = [os.path.join(args.data_dir,p) for p in args.test.split(",")]
        test_dataset = utils.open_dataset(test_paths,labels)
        predict(args,model,tokenizer,test_dataset)

if __name__ == "__main__":
    main()
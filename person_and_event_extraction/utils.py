import evaluate
import numpy as np
import datasets


def tokenize_and_align_labels(examples,tokenizer,encoding_args):
    tokenized_inputs = tokenizer(examples["tokens"], **encoding_args)# truncation=True, is_split_into_words=True, max_length=256)

    labels = []
    for i, label in enumerate(examples["bio_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

metric = evaluate.load("seqeval")
def compute_metrics(p,label_list):

    classes = label_list.copy()
    # remove O
    classes.remove("O")
    # remove the "B-" and "I-" prefixes
    classes = [c[2:] for c in classes]
    # remove duplicates
    classes = list(set(classes))

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

def open_dataset(path,label_dict):
    # the txt file is text with each line format is --> token\tlabel
    # each text is separated by a blank line
    paths = path.split(",")
    # if not endswith .txt then add .txt
    paths = [p if p.endswith(".txt") else p+".txt" for p in paths]
    data = []
    for path in paths:
        with open(path,"r") as f:
            text = f.read()
            data.extend(text.split("\n\n"))
    # remove blank lines
    data = [d for d in data if d != ""]
    data = [d.strip().split("\n") for d in data]
    # the columns : tokens, bio_tags
    dataset = {
        "tokens" : [],
        "bio_tags" : []
    }
    for d in data:
        tokens = []
        bio_tags = []
        for line in d:
            # print(line)
            token, bio_tag = line.split("\t")
            tokens.append(token)
            bio_tags.append(label_dict[bio_tag])
        dataset["tokens"].append(tokens)
        dataset["bio_tags"].append(bio_tags)
    
    return datasets.Dataset.from_dict(dataset)
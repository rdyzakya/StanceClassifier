import evaluate
import numpy as np

def tokenize_and_align_labels(examples,tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=256)

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
    # f1_person = results["person"]["f1"]
    # f1_event = results["event"]["f1"]
    # w_person = (1/results["person"]["number"]) / (1/results["person"]["number"] + 1/results["event"]["number"])
    # w_event = (1/results["event"]["number"]) / (1/results["person"]["number"] + 1/results["event"]["number"])
    # f1_weighted = f1_person * w_person + f1_event * w_event
    # print(results)
    # return {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": f1_weighted, # results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }
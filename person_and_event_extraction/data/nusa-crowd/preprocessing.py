import os
import pandas as pd
import datasets
import nltk
from nlp_id.tokenizer import Tokenizer

src_path = "./raw/data.csv"
output_path = "./interim/data.txt"
label_path = "../labels.txt"

# read dataset
df = pd.read_csv(src_path, sep=",", encoding="utf-8")
# read labels
with open(label_path, "r") as f:
    labels = f.read().splitlines()
    labels = {el: i for i, el in enumerate(labels)}

dataset = datasets.Dataset.from_pandas(df)

# sentence splitting
def sentence_tokenize(dataset):
    person = dataset["person"]
    event = dataset["event"]
    title = dataset["title"]
    stance = dataset["stance_final"]

    content = dataset["content"]

    # if the content length is zero, then the title held the content
    for i in range(len(content)):
        if content[i] == None:
            content[i] = title[i]
        else:
            if len(content[i]) == 0:
                content[i] = title[i]
    # sent tokenize
    sent_tokenized_content = [nltk.tokenize.sent_tokenize(el) for el in content]

    result = {
        "person" : [],
        "event" : [],
        "title" : [],
        "content" : [],
        "stance" : []
    }

    for i in range(len(sent_tokenized_content)):
        for j in range(len(sent_tokenized_content[i])):
            result["person"].append(person[i])
            result["event"].append(event[i])
            result["title"].append(title[i])
            result["content"].append(sent_tokenized_content[i][j])
            result["stance"].append(stance[i])

    return result

dataset = dataset.map(sentence_tokenize, batched=True, remove_columns=dataset.column_names)

def clean_special_char(text):
    text = text.encode('ascii','ignore').decode('utf-8')
    text = text.replace("’","'")
    text = text.replace("‘","'")
    text = text.replace("“",'"')
    text = text.replace("”",'"')
    text = text.replace("…","...")
    text = text.replace("—","-")
    text = text.replace("–","-")
    return text

# clean special char
dataset = dataset.map(lambda x: {'content' : clean_special_char(x['content'])})

tokenizer = Tokenizer()

def create_bio_label(row,labels=labels,tokenizer=tokenizer):
    text = row["content"]
    tokenized_text = tokenizer.tokenize(text)

    tokenized_person = tokenizer.tokenize(row["person"])
    joined_person = ' '.join(tokenized_person)

    tokenized_event = tokenizer.tokenize(row["event"])
    joined_event = ' '.join(tokenized_event)

    result_labels = {
        "person" : [],
        "event" : []
    }
    i = 0
    while i  < len(tokenized_text):
        window_person = tokenized_text[i:i+len(tokenized_person)]

        joined_window_person = ' '.join(window_person)
        if joined_window_person.lower() == joined_person.lower():
            added_labels = [labels["B-PERSON"]] + [labels["I-PERSON"]] * (len(window_person) - 1)
            result_labels["person"].extend(added_labels)
            i += len(window_person)
        else:
            result_labels["person"].append(labels["O"])
            i += 1

    i = 0
    while i  < len(tokenized_text):
        window_event = tokenized_text[i:i+len(tokenized_event)]

        joined_window_event = ' '.join(window_event)
        if joined_window_event.lower() == joined_event.lower():
            added_labels = [labels["B-EVENT"]] + [labels["I-EVENT"]] * (len(window_event) - 1)
            result_labels["event"].extend(added_labels)
            i += len(window_event)
        else:
            result_labels["event"].append(labels["O"])
            i += 1

    assert len(result_labels["person"]) == len(tokenized_text) == len(result_labels["event"])

    result_tags = []
    for i in range(len(tokenized_text)):
        new_token = result_labels["person"][i] + result_labels["event"][i]
        if new_token >= len(labels):
            # if overlapping then raise exception?
            raise Exception(f"No way : {text[:50]}")
        else:
            result_tags.append(new_token)
    return{ 'tokens' : tokenized_text, 'bio_tags' : result_tags}

tokenized_dataset = dataset.map(create_bio_label, batched=False, remove_columns=dataset.column_names)

labels_index = {v: k for k, v in labels.items()}

with open(output_path, "w") as f:
    for i in range(len(tokenized_dataset)):
        for j in range(len(tokenized_dataset[i]["tokens"])):
            # remove token with only space and O tag
            if tokenized_dataset[i]["tokens"][j].strip() != "" and tokenized_dataset[i]["bio_tags"][j] != 0:
                continue
            f.write(f"{tokenized_dataset[i]['tokens'][j]}\t{labels_index[tokenized_dataset[i]['bio_tags'][j]]}\n")
        f.write("\n")
from utils import open_dataset

path = "./data/nusa-crowd/interim/data.txt"

label_dict = {
    "O" : 0,
    "B-PERSON" : 1,
    "I-PERSON" : 2,
}

dataset = open_dataset(path,label_dict)

print(dataset)

print(dataset[0])
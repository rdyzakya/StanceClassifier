import argparse
import os

from sklearn.model_selection import train_test_split

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--file_name', type=str, default='data', help="seperated by comma")
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    return parser.parse_args()

def get_strat(dataset):
    # return 0 if the bio_tags unique is only 1 class, else 1
    strat = []
    for d in dataset:
        if len(set(d['bio_tags'])) == 1:
            strat.append(0)
        else:
            strat.append(1)
    return strat

def main():
    args = init_args()

    # add .txt to file name if not endswith .txt
    file_names = args.file_name.split(',')
    file_names = [file_name.strip() if file_name.strip().endswith('.txt') else file_name.strip() + '.txt' for file_name in file_names]

    dataset = []
    for fname in file_names:
        with open(os.path.join(args.data_dir,fname),'r',encoding="utf-8") as f:
            data = f.read().split("\n\n")
            data = [d.strip().split("\n") for d in data]
            # data = [{
            #     "tokens": [t.split("\t")[0] for t in d],
            #     "bio_tags": [t.split("\t")[1] for t in d]
            # } for d in data]
            for i_d in range(len(data)):
                d = data[i_d]
                tokens = []
                bio_tags = []
                for i_t in range(len(d)):
                    t = d[i_t]
                    if t == "":
                        continue
                    tokens.append(t.split("\t")[0])
                    bio_tags.append(t.split("\t")[1])
                data[i_d] = {
                    "tokens": tokens,
                    "bio_tags": bio_tags
                }

            dataset.extend(data)
    
    strat = get_strat(dataset)

    # split
    train, test = train_test_split(dataset, train_size=args.train_ratio, stratify=strat, random_state=42)

    # write to output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding="utf-8") as f:
        for d in train:
            for token, tag in zip(d['tokens'], d['bio_tags']):
                f.write(token + "\t" + tag + "\n")
            f.write("\n")
    
    with open(os.path.join(args.output_dir, 'test.txt'), 'w', encoding="utf-8") as f:
        for d in test:
            for token, tag in zip(d['tokens'], d['bio_tags']):
                f.write(token + "\t" + tag + "\n")
            f.write("\n")
    
    print("Successfully split data into train and test set")


if __name__ == "__main__":
    main()
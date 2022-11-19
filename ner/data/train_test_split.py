import argparse
import os

from sklearn.model_selection import train_test_split

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--file_name', type=str, default='data', help="seperated by comma")
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--with_val", action="store_true")
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

    # write to output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    strat = get_strat(dataset)

    # split
    train, val_test = train_test_split(dataset, train_size=args.train_ratio, stratify=strat, random_state=42)

    with open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding="utf-8") as f:
        for d in train:
            for token, tag in zip(d['tokens'], d['bio_tags']):
                f.write(token + "\t" + tag + "\n")
            f.write("\n")
    
    if args.with_val:
        args.test_ration = args.test_ratio / (1 - args.train_ratio)
        test, val = train_test_split(val_test, train_size=args.test_ratio, stratify=get_strat(val_test), random_state=42)
        with open(os.path.join(args.output_dir, 'val.txt'), 'w', encoding="utf-8") as f:
            for d in val:
                for token, tag in zip(d['tokens'], d['bio_tags']):
                    f.write(token + "\t" + tag + "\n")
                f.write("\n")
    else:
        test = val_test

    with open(os.path.join(args.output_dir, 'test.txt'), 'w', encoding="utf-8") as f:
        for d in test:
            for token, tag in zip(d['tokens'], d['bio_tags']):
                f.write(token + "\t" + tag + "\n")
            f.write("\n")
    
    print("Successfully split data into train and test set")


if __name__ == "__main__":
    main()
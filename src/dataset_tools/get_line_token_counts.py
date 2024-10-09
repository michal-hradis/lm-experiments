import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type = str,  required = True)
    parser.add_argument("--tokenizer-name", type = str, required = True)
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            print(f'{len(tokens)}\t{len(line)}\t{len(line.split())}\t{len(line)/len(line.split())}')

if __name__ == "__main__":
    main()
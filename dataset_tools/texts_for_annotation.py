import argparse
from glob import glob
from random import sample, shuffle
import os
import hashlib
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Takes text files, extract random lines from each and creates directories with text files for annotation."
                    "Part of the text files are shared between the directories to ensure that the same text is annotated by multiple annotators.")
    parser.add_argument("-i", "--input", required=True, help="Input directory.")
    parser.add_argument("-o", "--output", required=True, help="Output path.")
    parser.add_argument("--n", type=int, default=1, help="Number of lines to extract from each file.")
    parser.add_argument("--n-shared", type=int, default=1, help="Share every n-th file.")
    parser.add_argument("--n-directories", type=int, default=6, help="Number of directories to create.")
    parser.add_argument("--min-line-length", type=int, default=13, help="Minimum words in a line to consider it.")
    return parser.parse_args()


def main():
    args = parse_args()

    all_texts = []

    files = sorted(glob(os.path.join(args.input, '*.txt')))
    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if len(line.split()) >= args.min_line_length]
        if len(lines) <= args.n:
            continue
        for line in sample(lines, args.n):
            all_texts.append((file, line))



    shared_files = all_texts[::(args.n_shared * args.n_directories)]
    shared_files_set = set(shared_files)

    # remove shared files from all_texts
    all_texts = [text for text in all_texts if text not in shared_files_set]
    print(len(all_texts), len(shared_files))

    groups = [all_texts[i::args.n_directories] for i in range(args.n_directories)]

    # shuffle all groups and shared_files
    shuffle(shared_files)
    for group in groups:
        shuffle(group)
        print(len(group))


    # interleave shared files equidistantly in each group.
    # Every n-the file in a group should be from shared_files
    for group in groups:
        print(len(group))
        for i in range(0, len(group), args.n_shared):
            group.insert(i, shared_files[i // args.n_shared])
        print(len(group))

    for i, group in enumerate(groups):
        directory = os.path.join(args.output, f'{i}')
        os.makedirs(directory, exist_ok=True)
        print(len(group))
        for j, (file_name, line) in enumerate(group):
            file_hash = hashlib.md5(file_name.encode()).hexdigest()
            content_hash = hashlib.md5(line.encode()).hexdigest()
            output_file_name = f'{j:05d}_{file_hash}_{content_hash}.txt'
            with open(os.path.join(directory, output_file_name), 'w', encoding='utf-8') as f_out:
                f_out.write(line)


if __name__ == '__main__':
    main()


















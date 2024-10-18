import json
import argparse
import numpy as np


def parseargs():
    parser = argparse.ArgumentParser('Parse the jsonl files and extract the text.')
    parser.add_argument('-i', '--input', required=True, help='Input jsonl file.')
    parser.add_argument('-o', '--output', required=True, help='Output text file.')
    parser.add_argument('--min-length', type=int, default=2000, help='Minimum length of each line.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    line_lengths = []
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                data = json.loads(line)
                title = data['title']
                text = data['text']
                full_text = f'TEXT: {text} <eos> '
                if title:
                    full_text = f'TITLE: {title} ' + full_text
                full_text = full_text.replace('\n', ' <br> ')

                if len(full_text) < args.min_length:
                    continue

                f_out.write(full_text + '\n')
                line_lengths.append(len(full_text))

                if i % 10000 == 0:
                    print(f'Processed {i}, kept {len(line_lengths)}, avg length: {np.mean(line_lengths):.1f}, median length: {np.median(line_lengths):.1f}')


if __name__ == '__main__':
    main()
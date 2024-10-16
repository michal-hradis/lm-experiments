import json
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Parse the jsonl files and extract the text.')
    parser.add_argument('-i', '--input', required=True, help='Input jsonl file.')
    parser.add_argument('-o', '--output', required=True, help='Output text file.')
    parser.add_argument('--min-length', type=int, default=2000, help='Minimum length of each line.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                title = data['title']
                text = data['text']
                full_text = f'TEXT: {text}'
                if title:
                    full_text = f'TITLE: {title} ' + full_text
                full_text = full_text.replace('\n', ' <br> ')

                if len(full_text) >= args.min_length:
                    f_out.write(full_text + '\n')


if __name__ == '__main__':
    main()
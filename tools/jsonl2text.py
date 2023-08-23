import json
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Extract text from SemANT jsonl file and save it as a plain text corpus.')
    parser.add_argument('-o', '--output', required=True, help='Output text file.')
    parser.add_argument('-i', '--input', required=True, help='Input jsonl file.')
    parser.add_argument('--min-region-length', type=int, default=128, help='Minimum character length of a text region to be included in the output.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    with open(args.input, 'r', encoding='utf-8') as f:
        with open(args.output, 'w', encoding='utf-8') as g:
            for line in f:
                data = json.loads(line)
                for region in data['regions']:
                    if len(''.join(region['lines'])) < args.min_region_length:
                        continue

                    # remove trailing hyphens or add space at the end of each line
                    lines = [line[:-1] if line.endswith('-') else line + ' ' for line in region['lines']]
                    text = ''.join(lines)
                    g.write(text)
                    g.write('\n')


if __name__ == '__main__':
    main()
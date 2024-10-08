import argparse

def parseargs():
    parser = argparse.ArgumentParser('Take a text file and concatenate lines until each is at least a defined length.')
    parser.add_argument('-i', '--input', required=True, help='Input text file.')
    parser.add_argument('-o', '--output', required=True, help='Output text file.')
    parser.add_argument('--min-length', type=int, default=8000, help='Minimum length of each line.')
    parser.add_argument('--eol', default=' <EOL> ', help='End of line token.')
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(args.output, 'w', encoding='utf-8') as f:
        concatenated_line = []
        concatenated_line_length = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            concatenated_line.append(line)
            concatenated_line_length += len(line)
            if concatenated_line_length >= args.min_length:
                f.write(args.eol.join(concatenated_line) + '\n')
                concatenated_line = []
                concatenated_line_length = 0

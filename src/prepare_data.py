import numpy as np
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Convert text file with piece ids to numpy array.')
    parser.add_argument('-i', '--input', required=True, help='File wit piece ids to process.')
    parser.add_argument('-o', '--output', required=True, help='Output file.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    print('Reading')
    np_lines = np.zeros([0], dtype=np.uint16)
    lines = []
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            line = [int(word) for word in line.split()]
            lines.append(np.asarray([1] + line + [2], dtype=np.uint16))
            if (i + 1) % 100000 == 0:
                print('DONE', i)
                lines = np.concatenate(lines)
                np_lines = np.concatenate([np_lines, lines])
                lines = []

    if lines:
        lines = np.concatenate(lines)
        np_lines = np.concatenate([np_lines, lines])
    print(np_lines.dtype, np_lines.shape)
    np.save(args.output, np_lines)


if __name__ == '__main__':
    main()

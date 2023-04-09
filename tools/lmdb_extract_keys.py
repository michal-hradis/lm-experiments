import lmdb
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input LMDB.')
    return parser.parse_args()


def main():
    args = parse_args()
    env = lmdb.open(args.input, readonly=True, lock=False, readahead=True)
    with open(os.path.join(args.input, 'keys.txt'), 'w') as f:
        with env.begin(write=False) as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                print(key.decode(), file=f)


if __name__ == '__main__':
    main()

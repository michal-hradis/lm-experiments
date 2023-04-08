import lmdb
import sentencepiece as spm
import numpy as np
import argparse
import uuid


def parseargs():
    parser = argparse.ArgumentParser('Encode text from standard input with sentencepiece and store the result in lmdb.'
                                     ' Each line of the source file is treated as an independent document, encoded sep'
                                     'arately and stored in the lmdb with an unique uuid key.')
    parser.add_argument('-o', '--output', required=True, help='Output lmdb directory.')
    parser.add_argument('-m', '--model', help='Sentencepiece model.')
    parser.add_argument('--lowercase', action='store_true', help='Lowercase the input.')

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    sp = spm.SentencePieceProcessor(model_file=args.model, add_bos=True, add_eos=True, enable_sampling=True,
                                    nbest_size=-1, alpha=0.1)

    env = lmdb.open(args.output, map_size=1099511627776)
    with env.begin(write=True) as txn:
        with open(args.input, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if args.lowercase:
                    line = line.lower()
                line = sp.EncodeAsIds(line)
                key = str(uuid.uuid4()).encode('ascii')
                value = np.asarray(line, dtype=np.uint16)
                txn.put(key, value.tobytes())
                if (i + 1) % 100000 == 0:
                    print('DONE', i)


if __name__ == '__main__':
    main()


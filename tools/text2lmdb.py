import lmdb
import sentencepiece as spm
import numpy as np
import argparse
import sys
import hashlib
import zlib


def parseargs():
    parser = argparse.ArgumentParser('Encode text from standard input with sentencepiece and store the result in lmdb.'
                                     ' Each line of the source file is treated as an independent document, encoded sep'
                                     'arately and stored in the lmdb with an unique uuid key.')
    parser.add_argument('-o', '--output-trn', required=True, help='Output lmdb directory.')
    parser.add_argument('-v', '--output-val', required=True, help='Output lmdb directory.')
    parser.add_argument('--validation-ratio', type=int, default=100, help='Ratio of validation data.')
    parser.add_argument('-m', '--model', help='Sentencepiece model. If not specified, text will not be tokenized.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    if args.model:
        sp = spm.SentencePieceProcessor()
        sp.Load(args.model)
    else:
        sp = None

    env_trn = lmdb.open(args.output_trn, map_size=1000000000000)

    env_val = lmdb.open(args.output_val, map_size=1000000000000) if args.validation_ratio > 0 else None
    total_token_count = 0
    total_char_count = 0
    txn_trn = env_trn.begin(write=True)
    txn_val = env_val.begin(write=True) if env_val else None
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue
        key = hashlib.md5(line.encode('utf-8')).hexdigest()
        total_char_count += len(line)
        txn = txn_trn if i % args.validation_ratio else txn_val
        if sp:
            line = sp.EncodeAsIds(line)
            value = np.asarray(line, dtype=np.uint16)
            txn.put(key.encode('utf-8'), value.tobytes())
        else:
            # store the compressed line
            line = zlib.compress(line.encode('utf-8'))
            txn.put(key.encode('utf-8'), line)

        total_token_count += len(line)

        if (i + 1) % 10000 == 0:
            print(f'DONE {i}, chars: {total_char_count}, tokens: {total_token_count}, chars/token: {total_char_count / total_token_count:.2f}')
            # save the transaction
            txn_val.commit() if txn_val else None
            txn_trn.commit()
            txn_val = env_val.begin(write=True) if env_val else None
            txn_trn = env_trn.begin(write=True)


    txn.commit()
    env.close()


if __name__ == '__main__':
    main()

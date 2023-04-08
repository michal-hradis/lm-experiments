from torch.utils.data import Dataset
import lmdb
import numpy as np
import sentencepiece as spm
import torch
import zlib
import os


class TextDatasetLMDB(Dataset):
    """Pytorch dataset reading text from LMDB file and using sentencepiece tokenizer.

    The LMDB file can be created by tools/text2lmdb.py. Each lmdb entry is a utf-8 string representing a document.
    The dataset returns a tensor of token ids for the document. Token sequence is randomly cropped or padded to max_len.
    """

    def __init__(self, lmdb_path, sp_model_path, max_len=512, compressed=True, mask_prob=0.15, mask_whole_words=True, mask_token_id=3, random_token_prob=0.1):
        self.mask_prob = mask_prob
        self.mask_whole_words = mask_whole_words
        self.mask_token_id = mask_token_id
        self.random_token_prob = random_token_prob
        self.env = None
        self.txn = None
        self.lmdb_path = lmdb_path
        self.keys = {}
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        # get all tokens which split words
        self.word_split_token_ids = {i for i in range(self.sp.GetPieceSize()) if self.sp.IdToPiece(i).startswith('▁')}
        self.max_len = max_len
        self.pad_id = 0
        self.compressed = compressed
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        # get mapping from index to lmdb key
        for i, (key, _) in enumerate(self.txn.cursor()):
            self.keys[i] = key
            if len(self.keys) > 4000000:
                break
        #self.keys = {i: key for i, key in enumerate(self.txn.cursor().iternext(keys=True, values=False))}
        self.length = len(self.keys)

        self.name = os.path.basename(self.lmdb_path)

    def __len__(self):
        return self.length

    def do_mask_tokens(self, tokens):
        """ Mask random consecutive tokens with probability mask_prob and replace them with mask_token_id or a random token."""
        tokens = tokens.copy()
        masked_tokens = np.random.choice(len(tokens), size=int(len(tokens) * self.mask_prob + 0.5), replace=False)
        for i in masked_tokens:
            if np.random.rand() < self.random_token_prob:
                tokens[i] = np.random.randint(0, self.sp.GetPieceSize())
            else:
                tokens[i] = self.mask_token_id
        return tokens


    def do_mask_whole_words(self, tokens):
        """ Mask tokens with probability mask_prob and replace them with mask_token_id or random token.
        """
        tokens = tokens.copy()
        word_start_positions = [0] + [i for i, t in enumerate(tokens) if t in self.word_split_token_ids]
        if len(word_start_positions) < 2:
            # no word boundary in the sequence, just mask random tokens
            return self.do_mask_tokens(tokens)

        # mask whole words
        masked_tokens = np.random.choice(len(word_start_positions), size=int(len(word_start_positions) * self.mask_prob + 0.5), replace=False)
        for start in masked_tokens:
            pos = word_start_positions[start]
            if start < len(word_start_positions) - 1:
                end = word_start_positions[start + 1]
            else:
                end = len(tokens)
            for i in range(pos, end):
                if np.random.rand() < self.random_token_prob:
                    tokens[i] = np.random.randint(0, self.sp.GetPieceSize())
                else:
                    tokens[i] = self.mask_token_id
        return tokens

    def __getitem__(self, idx):
        #if self.env is None:

        item = self.txn.get(self.keys[idx])
        if self.compressed:
            item = zlib.decompress(item)
        tokens = self.sp.EncodeAsIds(item.decode('utf-8'))

        # randomly crop the sequence if too long but start at a word boundary
        if len(tokens) > self.max_len:
            word_start_positions = [0] + [i for i, t in enumerate(tokens) if
                                          t in self.word_split_token_ids and i < len(tokens) - self.max_len]
            if len(word_start_positions) < 2:
                # no word boundary in the sequence, just take the first max_len tokens
                tokens = tokens[:self.max_len]
            else:
                pos = np.random.choice(word_start_positions)
                tokens = tokens[pos:pos + self.max_len]

        if self.mask_whole_words:
            masked_tokens = self.do_mask_whole_words(tokens)
        else:
            masked_tokens = self.do_mask_tokens(tokens)

        # pad the sequence if too short
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
            masked_tokens = masked_tokens + [0] * (self.max_len - len(masked_tokens))

        return torch.LongTensor(tokens), torch.LongTensor(masked_tokens)

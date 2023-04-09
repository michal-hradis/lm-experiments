from learning_loop import LearningLoop
import argparse
from models.net_factory import net_factory
import torch
from dataset_text_lmdb import TextDatasetLMDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', required=True, help='Sentencepiece model.')
    parser.add_argument('--trn-dataset', required=True)
    parser.add_argument('--tst-data', action='append', default=[])
    parser.add_argument('--sequence-length', type=int, default=256)
    parser.add_argument('--net-config', default='{"type":"tansformer", "depth":6, "dim":"256"}')
    return parser


class Accuracy(torch.nn.Module):
    def __init__(self, mask_token_id=3):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, logits, targets):
        with torch.no_grad():
            predicted_tokens = logits.argmax(dim=-1)
            correct = (predicted_tokens == targets).float()
            correct[targets != self.mask_token_id] = 0
            correct_count = correct.sum()
            total_count = (targets == self.mask_token_id).sum()
            return correct_count / total_count

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, mask_token_id=0):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, logits, targets):
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.mask_token_id)
        return loss

import sentencepiece as spm

def main():
    parser = parse_args()
    parser = LearningLoop.add_params(parser)
    args = parser.parse_args()
    dataset_trn = TextDatasetLMDB(args.trn_dataset, args.tokenizer, args.sequence_length)
    dataset_tst = [TextDatasetLMDB(tst_data, args.tokenizer, args.sequence_length) for tst_data in args.tst_data]

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer)
    model = net_factory(args.net_config, token_count=tokenizer.GetPieceSize())
    print(model)

    LOSS = CrossEntropyLoss()
    METRICS = {'accuracy': Accuracy()}

    learning_loop = LearningLoop(args, model, loss=LOSS, metrics=METRICS, trn_dataset=dataset_trn,
                                 val_datasets=dataset_tst, tokenizer=tokenizer)
    learning_loop.run_training()


if __name__ == '__main__':
    main()
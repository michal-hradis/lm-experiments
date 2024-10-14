from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=32000, special_tokens=special_tokens, show_progress=True, limit_alphabet=700)
tokenizer.train(['/home/ihradis/projects/2021-12-10_LM/data/czech_all.txt'], trainer=trainer)

encoding = tokenizer.encode("Poslední jméno znělo Geoff Anstey. Dirk alespoň předpokládal, že bylo poslední, protože jako jediné nebylo tlustě přeškrtáno. Studoval škrtance a snažil")
print(encoding.tokens)
tokenizer.save("tokenizer.json")

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
wrapped_tokenizer.save_pretrained('./tokenizer')

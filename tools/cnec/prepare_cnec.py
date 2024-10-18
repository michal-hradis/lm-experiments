import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepares Czech Named Entity Corpus for training - removes all tags and prepares one sentence per line "
                    "without the tags and with the tags. It works with the plain format (e.g. Kolem <gt Asie> se rozkládá oceán.)")
    parser.add_argument("-i", "--input", required=True, help="Input file path.")
    parser.add_argument("-o", "--output", required=True, help="Output file path.")
    return parser.parse_args()


def remove_tags(text):
    words = [word for word in text.split() if not word.startswith('<')]
    return ' '.join(words).replace('>', '')

def prepare_prompt(text):
    text = text.strip()
    text = remove_tags(text)
    return f'Text: {text} <|EOS|> Pojmenované entity:"

def main():
    args = parse_args()
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                clean_text = remove_tags(line)
                f_out.write(f'Text: {clean_text} <|EOS|> Pojmenované entity: {line} <|EOS|>\n')


if __name__ == '__main__':
    main()
from datasets import load_dataset


# log in
# huggingface-cli login

dataset = load_dataset('BUT-FIT/BUT-LCC', split='train', streaming=True, use_auth_token=True)

for split in ['train', 'teset']:
    with open(f'BUT-LCC_{split}.txt', 'w') as f:
        for record in dataset:
            text = record['text']
            title = record['title']

            forward_order_string = f'Titulek: {title} <|EOS|> Text: {text} <|OER|>'
            backward_order_string = f'Text: {text} <|EOS|> Titulek: {title} <|OER|>'

            f.write(forward_order_string + '\n')
            f.write(backward_order_string + '\n')



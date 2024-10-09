from datasets import load_dataset

for split in ['validation', 'test']:
    dataset = load_dataset('hynky/czech_news_dataset_v2', split=split, streaming=True)
    with open(f'czech_news_dataset_{split}.txt', 'w') as f:
        dataset_iter = iter(dataset)
        for record in dataset_iter:
            date = record['date']
            category = record['category']
            headline = record['headline']
            keywords = ', '.join(record['keywords'])
            brief = record['brief']
            content = record['content']

            forward_order_string = f'Datum: {date} <|EOS|> Kategorie: {category} <|EOS|> Titulek: {headline} <|EOS|> Klíčová slova: {keywords} <|EOS|> Stručně: {brief} <|EOS|> Obsah: {content} <|OER|>'
            backward_order_string = f'Obsah: {content} <|EOS|> Stručně: {brief} <|EOS|> Klíčová slova: {keywords} <|EOS|> Titulek: {headline} <|EOS|> Kategorie: {category} <|EOS|> Datum: {date} <|OER|>'
            f.write(forward_order_string + '\n')
            f.write(backward_order_string + '\n')

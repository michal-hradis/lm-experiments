import sys


def main():
    ignored_tags = ('<s>', '<p>', '</s>')
    min_word_count = 128
    article = []
    article_word_count = 0
    words = []
    merge = False
    for line in sys.stdin:
        line = line.strip()
        if line == '</p>':
            article_word_count += len(words)
            article.append(' '.join(words + ['</p>']))
            words = ['<p>']
        elif line[:5] == '<doc ':
            words = ['<doc>'] + sorted([part for part in line.split() if "src=" in part or "title=" in part or "wiki_categories=" in part])[::-1]
            article = [' '.join(words)]
            article_word_count = 0
            words = ['<p>']
        elif line == '</doc>':
            if article_word_count > min_word_count:
                for x in article:
                    print(x)
            article = []
            article_word_count = 0
        elif line == '<g/>':
            merge = True
        elif line not in ignored_tags:
            if merge:
                merge = False
                if words:
                    words[-1] = words[-1] + line
            else:
                words.append(line)


if __name__ == '__main__':
    main()


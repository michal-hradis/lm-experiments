import sys


def main():
    ignored_tags = ('<s>', '<p>', '</s>', '</doc>')
    words = ['<p>']
    merge = False
    for line in sys.stdin:
        line = line.strip()
        if merge:
            merge = False
            if words:
                words[-1] = words[-1] + line
        elif line == '</p>':
            print(' '.join(words + ['</p>']))
            words = ['<p>']
        elif line[:5] == '<doc ':
            words = ['<doc>'] + [part for part in line.split() if "src=" in part or "title=" in part or "wiki_categories=" in part]
            print(' '.join(words))
            words = ['<p>']
        elif line == '<g/>':
            merge = True
        elif line not in ignored_tags:
            words.append(line)


if __name__ == '__main__':
    main()


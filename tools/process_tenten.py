import sys


def main():
    ignored_tags = ('<s>', '<p>', '</s>', '</doc>')
    words = []
    merge = False
    for line in open('c:\\projects\\lm-experiments\\data\\csTenTen17.txt.filepart', 'r', encoding='utf8'):#sys.stdin:
        line = line.strip()
        if merge:
            merge = False
            if words:
                words[-1] = words[-1] + line
        elif line == '</p>':
            print(' '.join(words))
            words = []
        elif line[:5] == '<doc ':
            print('<doc>')
        elif line == '<g/>':
            merge = True
        elif line not in ignored_tags:
            words.append(line)


if __name__ == '__main__':
    main()


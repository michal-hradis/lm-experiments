import sys
import lmdb
import uuid


class LMDBDataWriter:
    def __init__(self, path):
        self.env = lmdb.open(path, map_size=1099511627776)
        self.txn = self.env.begin(write=True)

    def write_new_ids(self, value):


        ""

def main():
    ignored_tags = ('<s>', '<p>', '</s>')
    min_word_count = 128
    article = []
    article_word_count = 0
    words = []
    merge = False
    for line in sys.stdin:
        line = line.strip()
        line_lower = line.lower()
        if line == '</p>':
            article_word_count += len(words)
            article.append(' '.join(words))
            words = ['<p/>']
        elif line[:5] == '<doc ':
            headers = []
            header = []
            for part in line.split():
                if "src=" in part or "title=" in part or "wiki_categories=" in part:
                    header = [part]
                if header and part[-1] == '"':
                    header = ' '.join(header).replace('|', ', ').replace('_', ' ')
                    headers.append(header)
                    header = None
                if header:
                    header.append(part)

            words = ['<doc>'] + sorted(headers)[::-1]
            article = [' '.join(words)]
            article_word_count = 0
            words = ['<p/>']
        elif line == '</doc>':
            if article_word_count > min_word_count:
                print(' '.join(article))
            article = []
            article_word_count = 0
        elif line in ['<g/>', '<g />']:
            merge = True
        elif line not in ignored_tags and line[:2] != '<s' and line_lower[:5] != "http:" and line_lower[:6] != "https:":
            if merge:
                merge = False
                if words:
                    words[-1] = words[-1] + line
            else:
                words.append(line)


if __name__ == '__main__':
    main()


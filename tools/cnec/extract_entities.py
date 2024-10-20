import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract entities from  CNEC plain annotation format for evaluation. The output format for example '<if Skody<gu Plzen>>',"
                    "assuming it is a sequence 1 (s1), is:\n"
                    "s1w1,s1w2\tif\tSkody Plzen\n"
                    "s1w2\tgu\tPlzen")
    parser.add_argument("-i", "--input", required=True, help="Input file path.")
    parser.add_argument("-o", "--output", required=True, help="Output file path.")
    return parser.parse_args()


def extract_entities(text: str, sequence_id: str) -> list[tuple[str, str, str]]:
    """
    Extracts entities from the text in the CNEC plain format.
    """
    # add spaces after each ">"
    text = text.replace(">", " >")
    text = text.replace("<", " <")
    words = text.split(" ")
    words = [w for w in words if w]
    text_words = [w for w in words if not (w.startswith("<") or w.startswith(">"))]

    entities = []
    stack = []
    word_position = 0
    for i, w in enumerate(words):
        if w.startswith("<"):
            tag = w[1:]
            stack.append((tag, word_position))
        elif w.startswith(">"):
            if not stack:
                logging.warning(f"Failed closing tag without opening tag: {w} at position {word_position} in sequence {sequence_id}")
                continue
            tag, start_position = stack.pop()
            entity = " ".join(text_words[start_position:word_position])
            entity_id = []
            for i in range(start_position, word_position):
                entity_id.append(f"s{sequence_id}w{i}")
            entity_id = ",".join(entity_id)
            entities.append((entity_id, tag, entity))
        else:
            word_position += 1

    if stack:
        logging.warning(f"Missing closing tags: {stack} in sequence {sequence_id}")
        for tag, start_position in stack:
            tag = tag[:1]
            entity = " ".join(text_words[start_position:])
            entity_id = []
            for i in range(start_position, word_position):
                entity_id.append(f"s{sequence_id}w{i}")
            entity_id = ",".join
            entities.append((entity_id, tag, entity))

    for tag, start_position in stack:
        entities.append((tag, start_position, word_position))

    return entities


def main():
    args = parse_args()
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for sequence_id, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                entities = extract_entities(line, sequence_id)
                for entity in entities:
                    f_out.write("\t".join(entity) + "\n")


if __name__ == '__main__':
    main()


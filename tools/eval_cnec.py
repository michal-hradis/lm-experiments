import numpy as np

def get_tags_as_numpy_array(tags: list[str], text: str) -> np.ndarray:
    """ Returns a numpy array of tags for a given text. The tags are in the CNEC format.

        Example: <P <pf Jan ><ps Novák>> žije v <gu Ústí nad <gh Labem>>.
    """

    # add spaces after each ">"
    text = text.replace(">", " >")
    words = text.split(" ")
    text_words = [w for w in words if not w.startswith("<") or w.startswith(">")]

    tag_array = np.zeros([len(tags), len(text_words)], dtype=bool)

    stack = []
    word_position = 0
    for i, w in enumerate(words):
        if w.startswith("<"):
            tag = w[1:]
            stack.append((tag, word_position))
        elif w.startswith(">"):
            tag, start_position = stack.pop()
            if tag in tags:
                tag_array[tags.index(tag), start_position:word_position] = True
            tag = tag[:1]
            if tag in tags:
                tag_array[tags.index(tag), start_position:word_position] = True
        else:
            word_position += 1

    for tag, start_position in stack:
        if tag in tags:
            tag_array[tags.index(tag), start_position:] = True

    return tag_array





def eval_cnec(gt: list[str], prediction: list[str]):
    """ Evaluates predictions on the CNEC dataset in "plain" format. The gt and predictions are in the same format.

        Example: <P <pf Jan><ps Novák>> žije v <gu Ústí nad <gh Labem>>.
    """

    ne_containers = ['P', "T", "A", "C"]
    ne_types = ['a', 'g', 'i', 'm', 'n', 'o', 'p', 't']
    # ah at az
    # gc gh gl gq gr gs gt gu g_
    # ia ic if io i_
    # me mi mn ms
    # nb nc ni no ns n_
    # oa oe om op or o_
    # pc pd pf pm pp ps p_
    # td tf th tm ty
    ne_subtypes = ['ah', 'at', 'az',
                   'gc', 'gh', 'gl', 'gq', 'gr', 'gs', 'gt', 'gu', 'g_',
                   'ia', 'ic', 'if', 'io', 'i_',
                   'me', 'mi', 'mn', 'ms',
                   'nb', 'nc', 'ni', 'no', 'ns', 'n_',
                   'oa', 'oe', 'om', 'op', 'or', 'o_',
                   'pc', 'pd', 'pf', 'pm', 'pp', 'ps', 'p_',
                   'td', 'tf', 'th', 'tm', 'ty']

    types_gt = get_tags_as_numpy_array(ne_types, gt)
    types_pred = get_tags_as_numpy_array(ne_types, prediction)

    subtypes_gt = get_tags_as_numpy_array(ne_subtypes, gt)
    subtypes_pred = get_tags_as_numpy_array(ne_subtypes, prediction)

    type_id_gt = np.argmax(types_gt, axis=0)
    type_id_pred = np.argmax(types_pred, axis=0)

    subtype_id_gt = np.argmax(subtypes_gt, axis=0)
    subtype_id_pred = np.argmax(subtypes_pred, axis=0)




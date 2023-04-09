import json
import logging
from swith_transformer import TextSwitchTransformer
from model_transformer import TextTransformer


def net_factory(config, token_count):
    if type(config) is str:
        config = json.loads(config)
    if config['type'].lower() == 'transformer':
        del config['type']
        model = TextTransformer(token_count=token_count, **config)
    elif config['type'].lower() == 'switch_transformer':
        del config['type']
        model = TextSwitchTransformer(token_count=token_count, **config)
    else:
        logging.error('Unknown net type "{config["type"]}".')
        exit(-1)
    return model
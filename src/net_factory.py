import json
import torch
import logging
from swith_transformer import TextSwitchTransformer





def net_factory(config, token_count):
    if type(config) is str:
        config = json.loads(config)
    if config['type'].lower() == 'transformer':
        del config['type']
        model = Transformer(token_count=token_count, **config)
    if config['type'].lower() == 'switch_transformer':
        del config['type']
        model = TextSwitchTransformer(token_count=token_count, **config)
    else:
        logging.error('Unknown net type "{config["type"]}".')
        exit(-1)
    return model
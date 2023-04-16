import json
import logging
from models.swith_transformer import TextSwitchTransformer
from models.model_transformer import TextTransformer
from models.transformer_multiscale import TextTransformerMultiscale
from models.conv_BERT import TextConvNetwork, TextMultiscaleConvNetwork

def net_factory(config, token_count, causal=False):
    if type(config) is str:
        config = json.loads(config)
    if config['type'].lower() == 'transformer':
        del config['type']
        model = TextTransformer(token_count=token_count, causal=causal, **config)
    elif config['type'].lower() == 'switch_transformer':
        del config['type']
        if causal:
            logging.error('Switch Transformer does not support causal mode.')
            exit(-1)
        model = TextSwitchTransformer(token_count=token_count, **config)
    elif config['type'].lower() == 'transformer_multiscale':
        del config['type']
        if causal:
            logging.error('Multiscale Transformer does not support causal mode.')
            exit(-1)
        model = TextTransformerMultiscale(token_count=token_count, **config)
    elif config['type'].lower() == 'conv':
        del config['type']
        if causal:
            logging.error('Convolutional Transformer does not support causal mode.')
            exit(-1)
        model = TextConvNetwork(token_count=token_count, **config)
    elif config['type'].lower() == 'multiscale_conv':
        if causal:
            logging.error('Multiscale Convolutional Transformer does not support causal mode.')
            exit(-1)
        del config['type']
        model = TextMultiscaleConvNetwork(token_count=token_count, **config)
    else:
        logging.error('Unknown net type "{config["type"]}".')
        exit(-1)
    return model
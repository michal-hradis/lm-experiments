# -*- coding: utf-8 -*-


import sentencepiece as spm
import numpy as np
import torch
import argparse
import math


def parseargs():
    parser = argparse.ArgumentParser('Train recurrent network LM.')
    parser.add_argument('-i', '--input_data', required=True, help='npy data file.')
    parser.add_argument('-v', '--vocabulary', required=True, help='Sentencepiece vocabulary.')

    parser.add_argument('--start-iteration', default=0, type=int, help='Start from this iteration.')

    parser.add_argument('--model-dim', default=512, type=int, help='Hidden state size.')
    parser.add_argument('--head-count', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--layers', default=6, type=int, help='Number of transformer blocks.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('--batch-length', default=64, type=int, help='Length of batch sequences.')
    parser.add_argument('--view-step', default=1000, type=int, help='How often we should test.')
    parser.add_argument('--max-iteration', default=100000, type=int, help='Hidden state size.')
    parser.add_argument('--learning-rate', default=0.0005, type=float, help='Learning rate.')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='Weight decay.')
    parser.add_argument('--conv', action='store_true', help='Use Conv net.')

    args = parser.parse_args()
    return args


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TextConvModel(torch.nn.Module):
    def __init__(self, token_count, model_dim=512, layers=6):
        super(TextConvModel, self).__init__()
        from conv_nets import ResidualConvBlock, ConvStack, CausalConvBlock

        self.token_count = token_count
        self.model_dim = model_dim
        self.layers = layers

        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.model_dim)

        module = ResidualConvBlock(self.model_dim, CausalConvBlock(self.model_dim))
        self.model = ConvStack(module, self.layers)

        self.decoder = torch.nn.Linear(
            in_features=self.model_dim,
            out_features=self.token_count,
            bias=True)

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        output = self.model(emb)
        logits = self.decoder(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

class TextTransformer(torch.nn.Module):
    def __init__(self, token_count, model_dim=512, head_count=8, layers=6):
        super(TextTransformer, self).__init__()

        self.token_count = token_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.layers = layers

        self.src_mask = generate_square_subsequent_mask(1024).to('cuda')
        #self.register_buffer('src_mask', self.src_mask)

        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.model_dim)
        self.position_encoder = PositionalEncoding(d_model=self.model_dim, dropout=0.025, max_len=1024)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.head_count, dim_feedforward=self.model_dim*4, dropout=0.025, batch_first=True)
        self.model = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        self.decoder = torch.nn.Linear(
            in_features=self.model_dim,
            out_features=self.token_count,
            bias=True)

        self.init_weights()

    def init_weights(self) -> None:
        pass
        #initrange = 0.002
        #self.embed.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        emb = self.position_encoder(emb)
        output = self.model(emb, self.src_mask[:emb.shape[1], :emb.shape[1]])
        logits = self.decoder(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def sample_characters(model, input_chars, char_count=100, temperature=1.0):
    sampled_chars = []
    with torch.no_grad():
        for i in range(char_count):
          log_probs = model(input_chars)[:, -1:, :]
          probs = torch.exp(log_probs) ** temperature
          probs /= probs.sum()
          distribution = torch.distributions.categorical.Categorical(probs=probs[0, 0, :])
          sampled_char = distribution.sample()
          input_chars = torch.cat([input_chars, sampled_char.reshape(1,1)], axis=1)
          sampled_chars.append(sampled_char.item())
    return sampled_chars


def print_samples(seed_text_ids, model, vocabulary):
    for ids in seed_text_ids:
        sampled_chars = sample_characters(model, ids, char_count=40, temperature=1.5)
        print('>>>>', vocabulary.decode_ids(ids.reshape(-1).tolist() + sampled_chars))


def main():
    args = parseargs()
    print(args)

    sp = spm.SentencePieceProcessor(args.vocabulary)
    text_ids = np.load(args.input_data).astype(np.int16)

    print(sp.decode(text_ids[:500].tolist()))
    print('|'.join([sp.decode(text_ids[i:i+1].tolist()) for i in range(100)]))

    try:
      device = torch.device('cuda')
    except:
      device = torch.device('cpu')
    print('Using device:', device)

    if args.conv:
        model = TextConvModel(sp.get_piece_size(), model_dim=args.model_dim, layers=args.layers)
    else:
        model = TextTransformer(sp.get_piece_size(), model_dim=args.model_dim, head_count=args.head_count, layers=args.layers).to(device=device)


    if args.start_iteration > 0:
        model.load_state_dict(torch.load(f'model_{args.start_iteration:07d}.pth', map_location=device))

    # Prepare and encode a prefix sequence.
    #seed_text = 'Některé zdroje Wikipedii kritizují pro její systematickou liberální (tj. liberálně-levicovou) '
    seed_texts = [
        'Umělý jazyk je jazyk, který byl vytvořen jedním člověkem nebo skupinou lidí. Významově pojem umělý jazyk není docela',
        '',
        'Jsou to vesměs suchobytné, více méně poléhavé nebo popínavé byliny',
        'V roce 1960 Malév, jako první letecká společnost z " východního bloku ", nasazuje',
        'Některé zdroje Wikipedii kritizují pro její systematickou liberální (tj. liberálně-levicovou) ']

    seed_text_ids = [np.array([1] + sp.encode(text)[:-2]) for text in seed_texts]

    # The prefix has to be a Torch tensor on the same device as the model.
    seed_text_ids = [torch.LongTensor(ids.reshape(1, -1)).to(device=device) for ids in seed_text_ids]

    #print_samples(seed_text_ids, model, sp)

    # Initial reading head positions.
    batch_heads = np.linspace(0, text_ids.size - args.batch_size*args.batch_length - 500, args.batch_size).astype(int)

    # The loss is coross-entropy.
    loss_fce = torch.nn.NLLLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_list = []
    acc_list = []


    for iteration in range(args.start_iteration, args.max_iteration):
        for head_id in range(args.batch_size):
            batch_heads[head_id] = np.random.randint(text_ids.size - args.batch_size*args.batch_length - 100)

        # Create training batch by stacking text from each reading head.
        batch_text = [text_ids[head:head+args.batch_length+1] for head in batch_heads]
        batch_text = np.stack(batch_text, axis=0)
        batch_text = torch.LongTensor(batch_text).to(device=device)

        network_input = batch_text[:, :-1]
        network_labels = batch_text[:, 1:]
        log_probs = model(network_input)

        transposed_log_probs = torch.transpose(log_probs, dim0=1, dim1=2)
        loss = loss_fce(transposed_log_probs, network_labels)

        # Compute how often the network gives highest probability to the correct
        # character.
        selected = torch.argmax(log_probs, axis=-1)
        acc = (batch_text[:, 1:] == selected).float().mean()
        acc = acc.item()  # We'd like store a single float value not a tensor connected to the huge computational graph.
        acc_list.append(acc)

        loss_list.append(loss.item()) # We'd like store a single float velue not a tensor connected to the huge computational graph.

        # Update the model.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress (loss, accuracy, generated text)
        if iteration % args.view_step == args.view_step -1:
            print()
            print(iteration, f'perplexity:{np.exp(np.average(loss_list)):0.2f} accuracy:{np.average(acc_list):0.3f}')
            print_samples(seed_text_ids, model, sp)
            loss_list = []
            acc_list = []
            torch.save(model.state_dict(), f'model_{iteration:07d}.pth')


if __name__ == '__main__':
    main()

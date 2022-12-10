# -*- coding: utf-8 -*-


import sentencepiece as spm
import numpy as np
import torch
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Train recurrent network LM.')
    parser.add_argument('-i', '--input_data', required=True, help='npy data file.')
    parser.add_argument('-v', '--vocabulary', required=True, help='Sentencepiece vocabulary.')

    parser.add_argument('--start-iteration', default=0, type=int, help='Start from this iteration.')

    parser.add_argument('--hidden-size', default=1024, type=int, help='Hidden state size.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('--batch-length', default=64, type=int, help='Length of batch sequences.')
    parser.add_argument('--view-step', default=1000, type=int, help='How often we should test.')
    parser.add_argument('--max-iteration', default=100000, type=int, help='Hidden state size.')
    parser.add_argument('--learning-rate', default=0.0005, type=float, help='Learning rate.')
    parser.add_argument('--weight-decay', default=0.0002, type=float, help='Weight decay.')

    args = parser.parse_args()
    return args


class Text_RNN(torch.nn.Module):
    def __init__(self, char_count, hidden_size=512):
        super(Text_RNN, self).__init__()

        self.char_embbedding_size = 320
        self.output_sqeeze_size = 320
        self.hidden_size = hidden_size

        self.embed = torch.nn.Embedding(char_count, embedding_dim=self.char_embbedding_size)

        self.rnn = torch.nn.LSTM(
            input_size=self.char_embbedding_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True)

        self.sqeeze_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_sqeeze_size)

        self.output_layer = torch.nn.Linear(
            in_features=self.output_sqeeze_size,
            out_features=char_count)

    def forward(self, input_chars, state=None):
        emb = self.embed(input_chars)
        output, out_state = self.rnn(emb, state)
        sqeezed = self.sqeeze_layer(output)
        sqeezed = torch.nn.functional.tanh(sqeezed)
        logits = self.output_layer(sqeezed)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, out_state


def sample_characters(model, input_chars, char_count=100, temperature=1.0, state=None):
    sampled_chars = []
    with torch.no_grad():
        for i in range(char_count):
          log_probs, state = model(input_chars, state=state)
          log_probs = log_probs[:, -1:, :]
          probs = torch.exp(log_probs) ** temperature
          probs /= probs.sum()
          distribution = torch.distributions.categorical.Categorical(probs=probs[0, 0, :])
          sampled_char = distribution.sample()
          sampled_chars.append(sampled_char.item())
          input_chars = sampled_char.reshape(1, 1)
    return sampled_chars


def print_samples(seed_text_ids, model, vocabulary):
    for ids in seed_text_ids:
        sampled_chars = sample_characters(model, ids, char_count=300, temperature=1.5)
        print('>>>>', vocabulary.decode_ids(ids.reshape(-1).tolist() + sampled_chars))


def main():
    args = parseargs()

    sp = spm.SentencePieceProcessor(args.vocabulary)
    text_ids = np.load(args.input_data).astype(np.int16)

    print(sp.decode(text_ids[:500].tolist()))
    print('|'.join([sp.decode(text_ids[i:i+1].tolist()) for i in range(100)]))

    try:
      device = torch.device('cuda')
    except:
      device = torch.device('cpu')
    print('Using device:', device)

    model = Text_RNN(sp.get_piece_size(), hidden_size=args.hidden_size).to(device=device)
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

    print_samples(seed_text_ids, model, sp)

    # Initial reading head positions.
    batch_heads = np.linspace(0, text_ids.size - args.batch_size*args.batch_length - 500, args.batch_size).astype(int)

    # Initial state is all zeros (behavior of LSTM when state is None).
    model_state = None

    # The loss is coross-entropy.
    loss_fce = torch.nn.NLLLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_list = []
    acc_list = []

    for iteration in range(args.start_iteration, args.max_iteration):

        # Advance reading heads to the next position.
        batch_heads += args.batch_length
        # This head reset should do nothing. Just in case ...
        batch_heads[batch_heads > text_ids.size - 2*args.batch_length] = np.random.randint(args.batch_length)

        # Create training batch by stacking text from each reading head.
        batch_text = [text_ids[head:head+args.batch_length+1] for head in batch_heads]
        batch_text = np.stack(batch_text, axis=0)
        batch_text = torch.LongTensor(batch_text).to(device=device)

        network_input = batch_text[:, :-1]
        network_labels = batch_text[:, 1:]
        log_probs, model_state = model(network_input, model_state)

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

        # Keep the LSTM state for the nest iteration, but disconect (detach) it from the graph.
        model_state = [state.detach() for state in model_state]

        # Reset several reading heads and corresponging LSTM states.
        for i in range(0, args.batch_size, 32):
            head_id = (iteration + i) % args.batch_size
            model_state[0][:, head_id, ...] = 0
            model_state[1][:, head_id, ...] = 0
            batch_heads[head_id] = np.random.randint(text_ids.size - args.batch_size*args.batch_length - 100)

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

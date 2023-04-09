# -*- coding: utf-8 -*-


import sentencepiece as spm
import numpy as np

import argparse
import math
import time
import torch


def parseargs():
    parser = argparse.ArgumentParser('Train recurrent network LM.')
    parser.add_argument('-i', '--input_data', required=True, help='npy data file.')
    parser.add_argument('-v', '--vocabulary', required=True, help='Sentencepiece vocabulary.')
    parser.add_argument('--data-limit', default=-1, type=int, help='Start from this iteration.')

    parser.add_argument('--start-iteration', default=0, type=int, help='Start from this iteration.')

    parser.add_argument('--model-dim', default=512, type=int, help='Hidden state size.')
    parser.add_argument('--head-count', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--layers', default=6, type=int, help='Number of transformer blocks.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('--batch-length', default=64, type=int, help='Length of batch sequences.')
    parser.add_argument('--view-step', default=1000, type=int, help='How often we should test.')
    parser.add_argument('--max-iteration', default=100000, type=int, help='Hidden state size.')
    parser.add_argument('--learning-rate', default=0.0005, type=float, help='Learning rate.')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='Weight decay.')
    parser.add_argument('--experts', default=1, type=int, help='Number of experts.')
    parser.add_argument('--conv', action='store_true', help='Use Conv net.')
    parser.add_argument('--warmup', default=2000, type=int, help='Linear warmup iterations starting from 0.')
    parser.add_argument('--gradient-clip', default=2000, type=float, help='Per-value gradinet clipping value.')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout.')
    parser.add_argument('--start-conv-count', default=0, type=int, help='Number of convolutional layers at the beginning of the model.')
    parser.add_argument('--end-conv-count', default=0, type=int, help='Number of convolutional layers at the end of the model.')

    args = parser.parse_args()
    return args


def sample_characters(model, input_chars, char_count=100, temperature=1.0):
    sampled_chars = []
    with torch.no_grad():
        for i in range(char_count):
          log_probs, counts, route_prob, n_dropped, route_prob_max = model(input_chars)
          #print(counts)
          #print(n_dropped)
          log_probs = log_probs[:, -1:, :]
          probs = torch.exp(log_probs) ** temperature
          probs /= probs.sum()
          distribution = torch.distributions.categorical.Categorical(probs=probs[0, 0, :])
          sampled_char = distribution.sample()
          input_chars = torch.cat([input_chars, sampled_char.reshape(1,1)], axis=1)
          sampled_chars.append(sampled_char.item())
    return sampled_chars


def print_samples(seed_text_ids, model, vocabulary):
    model.eval()
    for ids in seed_text_ids:
        sampled_chars = sample_characters(model, ids, char_count=40, temperature=1.5)
        print('>>>>', vocabulary.decode_ids(ids.reshape(-1).tolist() + sampled_chars))
    model.train()




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

    #if args.conv:
    #    model = TextConvModel(sp.get_piece_size(), model_dim=args.model_dim, layers=args.layers)
    #else:
    #    model = TextTransformer(sp.get_piece_size(), model_dim=args.model_dim, head_count=args.head_count, layers=args.layers,
    #                            start_conv=args.start_conv_count, end_conv=args.end_conv_count).to(device=device)

    model = TextSwitchTransformer(sp.get_piece_size(),
                                  model_dim=args.model_dim, head_count=args.head_count, layers=args.layers,
                                  n_experts=args.experts).to(device=device)

    print(model)

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

    seed_text_ids = [np.array([1] + sp.encode(text)[:-2]) for text in seed_texts] + [np.array([1])]

    # The prefix has to be a Torch tensor on the same device as the model.
    seed_text_ids = [torch.LongTensor(ids.reshape(1, -1)).to(device=device) for ids in seed_text_ids]

    if args.start_iteration > 0:
        model.load_state_dict(torch.load(f'model_{args.start_iteration:07d}.pth', map_location=device))
        #model.eval()
        #print_samples(seed_text_ids, model, sp)
        #model.train()


    #print_samples(seed_text_ids, model, sp)

    # Initial reading head positions.
    batch_heads = np.linspace(0, text_ids.size - args.batch_size*args.batch_length - 500, args.batch_size).astype(int)

    # The loss is coross-entropy.
    loss_fce = torch.nn.NLLLoss().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_list = []
    acc_list = []
    time_list = []


    for iteration in range(args.start_iteration, args.max_iteration):
        for head_id in range(args.batch_size):
            batch_heads[head_id] = np.random.randint(text_ids.size - args.batch_size*args.batch_length - 100)

        # Create training batch by stacking text from each reading head.
        batch_text = [text_ids[head:head+args.batch_length+1] for head in batch_heads]
        batch_text = np.stack(batch_text, axis=0)
        batch_text = torch.LongTensor(batch_text).to(device=device)

        network_input = batch_text[:, :-1]
        network_labels = batch_text[:, 1:]
        t1 = time.time()
        log_probs, counts, route_prob, n_dropped, route_prob_max  = model(network_input)
        print(n_dropped)

        total = counts.sum(dim=-1, keepdims=True)
        # Fraction of tokens routed to each expert
        # $$f_i = \frac{1}{T} \sum_{x \in \mathscr{B}} \mathbf{1} \{ \mathop{argmax} p(x), i \}$$
        # $f_i$ is the count of tokens where the argmax of $p(x)$ is equal to $i$.
        route_frac = counts / total
        # Mean routing probability
        # $$P_i = \frac{1}{T} \sum_{x \in \mathscr{B}} p_i (x)$$
        route_prob = route_prob / total
        # Load balancing loss
        # $$\mathscr{L} = N \sum_{i=1}^N f_i \cdot P_i$$
        # $\mathscr{L}$ is the loss for a single layer and here we are
        # taking the sum of losses across all layers.
        load_balancing_loss = args.experts * (route_frac * route_prob).sum()

        transposed_log_probs = torch.transpose(log_probs, dim0=1, dim1=2)
        cross_entropy_loss = loss_fce(transposed_log_probs, network_labels)

        # Combined loss.
        # The load balancing loss is multiplied by a coefficient $\alpha$ which is
        # set to something small like $\alpha = 0.01$.
        loss = cross_entropy_loss + 0.01 * load_balancing_loss

        print(loss.item(), cross_entropy_loss.item(), (0.01 * load_balancing_loss).item())
        # Update the model.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        time_list.append(t2-t1)

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
        if args.gradient_clip > 0:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gradient_clip)
        optimizer.step()

        # print progress (loss, accuracy, generated text)
        if iteration % args.view_step == args.view_step -1:
            print()
            print(iteration, f'perplexity:{np.exp(np.average(loss_list)):0.2f} accuracy:{np.average(acc_list):0.3f} {1/np.mean(time_list):0.1f}b/s')
            print_samples(seed_text_ids, model, sp)
            loss_list = []
            acc_list = []
            if iteration != args.start_iteration:
                torch.save(model.state_dict(), f'model_{iteration:07d}.pth')


if __name__ == '__main__':
    main()

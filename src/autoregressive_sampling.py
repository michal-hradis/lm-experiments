import torch


def sample_characters(model, input_chars, char_count=100, temperature=1.0):
    if input_chars.ndim == 1:
        input_chars = input_chars.reshape(1, -1)
    sampled_chars = []
    with torch.no_grad():
        for i in range(char_count):
          log_probs = model(input_chars)
          #log_probs, counts, route_prob, n_dropped, route_prob_max
          log_probs = log_probs[:, -1:]
          probs = torch.exp(log_probs) ** temperature
          probs /= probs.sum()
          distribution = torch.distributions.categorical.Categorical(probs=probs)
          sampled_char = distribution.sample()
          input_chars = torch.cat([input_chars, sampled_char], axis=1)
          sampled_chars.append(sampled_char.item())
    return sampled_chars


def print_samples(seed_text_ids, model, vocabulary):
    model.eval()
    for ids in seed_text_ids:
        sampled_chars = sample_characters(model, ids, char_count=40, temperature=1.5)
        print('>>>>', vocabulary.decode_ids(ids.reshape(-1).tolist() + sampled_chars))
    model.train()

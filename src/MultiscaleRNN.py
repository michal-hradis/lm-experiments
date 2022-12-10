import torch


def upsample(x):
    return torch.cat([x, torch.zeros_like(x)], dim=2).reshape(x.shape[0], x.shape[1]*2, x.shape[2])


class TextMultiscaleRNN(torch.nn.Module):
    def __init__(self, char_count, hidden_size=512, input_embbedding_size=512, depth=3):
        super(TextMultiscaleRNN, self).__init__()

        self.input_embbedding_size = 256
        self.hidden_size = hidden_size

        self.embed = torch.nn.Embedding(char_count, embedding_dim=self.input_embbedding_size)


        self.down_layers = []
        self.up_layers = []
        for i in range(depth):
            if i == 0:
                input_dim = self.input_embbedding_size
            else:
                input_dim = int(self.hidden_size * 2 ** ((i - 1) / 2.0))
            print(input_dim)
            self.down_layers.append(
                torch.nn.LSTM(
                    input_size=input_dim, hidden_size=int(self.hidden_size * 2 ** (i/2.0)), num_layers=1, batch_first=True)
            )

            self.up_layers.append(
                torch.nn.LSTM(
                    input_size=int(self.hidden_size * 2 ** (i/2.0) + self.hidden_size * 2 ** ((i+1)/2.0)),
                    hidden_size=int(self.hidden_size * 2 ** (i/2.0)), num_layers=1, batch_first=True)
            )

        self.down_layers.append(
            torch.nn.LSTM(
                input_size=int(self.hidden_size * 2 ** ((depth-1) / 2.0)),
                hidden_size=int(self.hidden_size * 2 ** (depth / 2.0)), num_layers=1, batch_first=True)
        )
        self.up_layers = self.up_layers[::-1]

        self.down_layers = torch.nn.ModuleList(self.down_layers)
        self.up_layers  = torch.nn.ModuleList(self.up_layers )


        self.output_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=char_count)

    def forward(self, input_chars, states=None):
        if states is None:
            states = [None] * (len(self.up_layers) + len(self.down_layers))

        emb = self.embed(input_chars)
        temp = emb
        bypass = []
        output_states = []
        for layer, state in zip(self.down_layers, states):
            if temp is not emb:
                temp = temp[:, ::2]
            temp, out_state = layer(temp, state)
            bypass.append(temp)
            output_states.append(out_state)
        bypass = bypass[:-1]

        for layer, b, state in zip(self.up_layers, bypass[::-1], states[len(self.down_layers):]):
            temp = upsample(temp)
            temp = torch.cat([temp, b], dim=2)
            temp, out_state = layer(temp, state)
            output_states.append(out_state)

        logits = self.output_layer(temp)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, output_states


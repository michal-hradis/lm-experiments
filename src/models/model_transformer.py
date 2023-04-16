import torch
from models.conv_nets import ResidualConvBlock, ConvStack, ConvBlock
from models.model_common import generate_square_subsequent_mask, PositionalEncoding


class TextConvModel(torch.nn.Module):
    def __init__(self, token_count, model_dim=512, layers=6):
        super(TextConvModel, self).__init__()

        self.token_count = token_count
        self.model_dim = model_dim
        self.layers = layers

        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.model_dim)

        module = ResidualConvBlock(self.model_dim, ConvBlock(self.model_dim))
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
    def __init__(self, token_count, model_dim=512, head_count=8, layers=6, start_conv=0, end_conv=0, dropout=0.025, causal=False):
        super(TextTransformer, self).__init__()

        self.token_count = token_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.layers = layers
        dropout = 0
        self.dropout = dropout
        device = 'cpu'

        self.src_mask = generate_square_subsequent_mask(1024).to(device) if causal else torch.zeros((1024, 1024)).to(device)
        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.model_dim)

        if start_conv:
            module = ResidualConvBlock(self.model_dim, ConvBlock(self.model_dim, causal=causal))
            self.start_conv = ConvStack(module, start_conv)
        else:
            self.start_conv = None

        self.position_encoder = PositionalEncoding(d_model=self.model_dim, dropout=dropout, max_len=1024)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.head_count, dim_feedforward=self.model_dim*4, dropout=dropout, batch_first=True)

        self.model = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        if end_conv:
            module = ResidualConvBlock(self.model_dim, ConvBlock(self.model_dim, causal=causal))
            self.end_conv = ConvStack(module, end_conv)
        else:
            self.end_conv = None

        self.decoder = torch.nn.Linear(
            in_features=self.model_dim,
            out_features=self.token_count,
            bias=True)

        self.input_layer_norm = torch.nn.LayerNorm(self.model_dim)

        #self.init_weights()

    #def init_weights(self) -> None:
    #    pass
        #initrange = 0.002
        #self.embed.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        emb = self.start_conv(emb) if self.start_conv else emb
        # print energy in embeddings
        emb = self.position_encoder(emb)
        #emb = self.input_layer_norm(emb)
        if self.src_mask is not None:
            output = self.model(emb, self.src_mask[:emb.shape[1], :emb.shape[1]])
        else:
            output = self.model(emb)
        output = self.end_conv(output) if self.end_conv else output
        logits = self.decoder(output)

        #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return logits

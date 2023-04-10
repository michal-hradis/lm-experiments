import torch
import copy


class PureConvBlock(torch.nn.Module):
    def __init__(self, model_dim, inner_dim=None, groups=[8, 1], kernel_size=[7, 1]):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = [1, 1]
        self.groups = groups
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim else model_dim

        self.layer1 = torch.nn.Conv1d(model_dim, self.inner_dim, self.kernel_size[0],
                                     stride=1, dilation=self.dilation[0], groups=self.groups[0],
                                     padding=(self.kernel_size[0] - 1) * self.dilation[0] // 2)

        self.layer2 = torch.nn.Conv1d(self.inner_dim, model_dim, self.kernel_size[1],
                                     stride=1, dilation=self.dilation[1], groups=self.groups[1],
                                     padding=(self.kernel_size[1] - 1) * self.dilation[1] // 2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.selu(x, inplace=True)
        x = self.layer2(x)
        x = torch.nn.functional.selu(x, inplace=True)
        return x


class ResidualConvBlock(torch.nn.Module):
    def __init__(self, model_dim, conv_block):
        super(ResidualConvBlock, self).__init__()
        self.comp_block = conv_block
        self.layer_norm_eps: float = 1e-5
        self.norm = torch.nn.LayerNorm(model_dim, eps=self.layer_norm_eps)

    def forward(self, x):
        x = x + self.norm(self.comp_block(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.comp_block(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TextConvNetwork(torch.nn.Module):
    def __init__(self, token_count, base_dims=256, expansion_dims=1024, groups=[8, 1], kernel_size=[5, 5], stage_count=4, dropout=0.025):
        super().__init__()
        self.token_count = token_count
        self.base_dims = base_dims
        self.stage_count = stage_count
        self.dropout = dropout
        self.expansion_dims = expansion_dims
        self.groups = groups
        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.base_dims)
        # scale embedding values
        with torch.no_grad():
            self.embed.weight.data *= 0.01
        stage = PureConvBlock(self.base_dims, self.expansion_dims, groups=self.groups, kernel_size=kernel_size)
        stage = ResidualConvBlock(self.base_dims, stage)
        self.stages = _get_clones(stage, self.stage_count)

        self.decoder = torch.nn.Linear(
            in_features=self.base_dims,
            out_features=self.token_count,
            bias=True)

        # initialize weights
        with torch.no_grad():
            self.decoder.weight.uniform_(-0.1, 0.1)
            self.decoder.bias.zero_()

        self.counter = 0

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        emb = emb.permute(0, 2, 1)
        self.counter += 1
        if self.counter % 100 == 0:
            print(f"Emb {0} mean: {emb.mean()}, std: {emb.std()}")

        for i, stage in enumerate(self.stages):
            emb = stage(emb)
            # print emb statistics mean std
            if self.counter % 100 == 1:
                print(f"Emb {i+1} mean: {emb.mean()}, std: {emb.std()}")
        emb = emb.permute(0, 2, 1)
        logits = self.decoder(emb)

        #print(f"Last layer weights : {self.decoder.weight.mean()}, std: {self.decoder.weight.std()}")
        if self.counter % 100 == 0:
            if self.decoder.bias is not None:
                print(f"Last layer bias : {self.decoder.bias.mean()}, std: {self.decoder.bias.std()}")
            if self.counter == 1:
                print(f"Logits mean: {logits.mean()}, std: {logits.std()}")

        #print(list(self.embed.parameters()))
        return logits


def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

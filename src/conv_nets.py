import torch
import copy


class ConvBlock(torch.nn.Module):
    def __init__(self, model_dim, causal=False):
        super(ConvBlock, self).__init__()

        self.kernel_size = 5
        self.dilation = 1
        self.groups = 1
        self.model_dim = model_dim
        self.causal = causal
        if self.causal:
            self.layer1 = torch.nn.Conv1d(model_dim, model_dim, self.kernel_size,
                                         stride=1, dilation=self.dilation, groups=1,
                                         padding=(self.kernel_size - 1) * self.dilation)

            self.layer2 = torch.nn.Conv1d(model_dim, model_dim, self.kernel_size,
                                         stride=1, dilation=self.dilation, groups=1,
                                         padding=(self.kernel_size - 1) * self.dilation)
        else:
            self.layer1 = torch.nn.Conv1d(model_dim, model_dim, self.kernel_size,
                                         stride=1, dilation=self.dilation, groups=1,
                                         padding=(self.kernel_size - 1) * self.dilation // 2)

            self.layer2 = torch.nn.Conv1d(model_dim, model_dim, self.kernel_size,
                                         stride=1, dilation=self.dilation, groups=1,
                                         padding=(self.kernel_size - 1) * self.dilation // 2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = x[:, :, 0:-(self.kernel_size-1)*self.dilation] if self.causal else x
        x = self.layer2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = x[:, :, 0:-(self.kernel_size-1)*self.dilation] if self.causal else x
        return x


class ResidualConvBlock(torch.nn.Module):
    def __init__(self, model_dim, conv_block):
        super(ResidualConvBlock, self).__init__()
        self.comp_block = conv_block
        self.layer_norm_eps: float = 1e-5
        self.norm = torch.nn.LayerNorm(model_dim, eps=self.layer_norm_eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.comp_block(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x


class ConvStack(torch.nn.Module):
    def __init__(self, conv_block, block_count):
        super(ConvStack, self).__init__()
        self.layers = _get_clones(conv_block, block_count)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

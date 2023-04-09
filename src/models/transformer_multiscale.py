import torch
from conv_nets import ResidualConvBlock, ConvStack, ConvBlock
from src.models.model_common import PositionalEncoding


def compute_conv1DTransposed_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


class TransformerBackbone(torch.nn.Module):
    def __init__(self, start_stage=0, base_dims=256, base_heads=4, stage_layers=2, stage_count=4, stage_subsampling=4, dropout=0.025):
        super().__init__()
        self.base_dims = base_dims
        self.base_heads = base_heads
        self.stage_layers = stage_layers
        self.stage_count = stage_count
        self.stage_subsampling = stage_subsampling
        self.dropout = dropout
        self.start_stage = start_stage

        self.positional_encoders = [
            PositionalEncoding(d_model=self.base_dims * 2 ** stage, dropout=dropout, max_len=512)
            for stage in range(start_stage, self.stage_count)]

        self.stage_modules = []
        for stage in range(start_stage, self.stage_count):

            if stage == 0:
                encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.base_dims * 2 ** stage,
                                                                 nhead=self.base_heads * 2 ** stage,
                                                                 dim_feedforward=self.base_dims * 2 ** stage * 4,
                                                                 dropout=dropout, batch_first=True)
                stage = torch.nn.ModuleList([torch.nn.TransformerEncoder(encoder_layer, num_layers=self.stage_layers)])
                self.stage_modules.append(stage)
            else:
                conv_upsample = torch.nn.Conv1d(self.base_dims * 2 ** (stage - 1), self.base_dims * 2 ** stage,
                                                kernel_size=3, padding=1)
                activation = torch.nn.ReLU()
                norm = torch.nn.LayerNorm(self.base_dims * 2 ** (stage))
                encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.base_dims * 2 ** stage,
                                                                 nhead=self.base_heads * 2 ** stage,
                                                                 dim_feedforward=self.base_dims * 2 ** stage * 4,
                                                                 dropout=dropout, batch_first=True)
                transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.stage_layers)
                stage = torch.nn.ModuleList([conv_upsample, activation, norm, transformer])
                self.stage_modules.append(stage)
        self.stage_modules = torch.nn.ModuleList(self.stage_modules)

    def forward(self, data):
        stage_results = []
        for i, stage in enumerate(self.stage_modules):
            if i > 0:
                data = data.permute(0, 2, 1)
                data = torch.nn.functional.max_pool1d(data, kernel_size=self.stage_subsampling,
                                                      stride=self.stage_subsampling)
                data = stage[0](data)
                data = stage[1](data)
                data = data.permute(0, 2, 1)
                data = stage[2](data)
                data = self.positional_encoders[i](data)
            data = stage[-1](data)
            stage_results.append(data)
        return stage_results


class TransformerNeck(torch.nn.Module):
    def __init__(self, base_dims=256, base_heads=4, stage_layers=2, stage_count=4, stage_subsampling=4, dropout=0.025):
        super().__init__()

        self.base_dims = base_dims
        self.base_heads = base_heads
        self.stage_layers = stage_layers
        self.stage_count = stage_count
        self.stage_subsampling = stage_subsampling
        self.dropout = dropout

        self.stage_modules = []
        for stage in range(self.stage_count):
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.base_dims * 2 ** stage,
                                                     nhead=self.base_heads * 2 ** stage,
                                                     dim_feedforward=self.base_dims * 2 ** stage * 4,
                                                     dropout=dropout, batch_first=True)
            transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.stage_layers)
            if stage == 0:
                stage = transformer
            else:
                norm = torch.nn.LayerNorm(self.base_dims * 2 ** stage)
                activation = torch.nn.ReLU()
                conv_upsample_squeeze = torch.nn.ConvTranspose1d(self.base_dims * 2 ** stage, self.base_dims * 2 ** (stage - 1),
                                                                    kernel_size=self.stage_subsampling, padding=0, stride=self.stage_subsampling)

                stage = torch.nn.ModuleList([transformer, norm, activation, conv_upsample_squeeze])
            self.stage_modules.append(stage)
        print(self.stage_modules)
        self.stage_modules = torch.nn.ModuleList(self.stage_modules)

    def forward(self, encoder_features):
        data = None
        for i, stage, features in list(zip(range(len(self.stage_modules)), self.stage_modules, encoder_features))[::-1]:
            if data is None:
                data = features
            else:
                data = data + features


            if i > 0:
                data = stage[0](data)
                data = stage[1](data)
                data = stage[2](data)
                data = data.permute(0, 2, 1)
                data = stage[3](data)
                data = data.permute(0, 2, 1)
            else:
                data = stage(data)

        return data


class TextTransformerMultiscale(torch.nn.Module):
    def __init__(self, token_count, base_dims=256, base_heads=4, stage_layers=2, stage_count=4, stage_subsampling=4, start_conv=0, end_conv=0, dropout=0.025):
        super().__init__()

        self.token_count = token_count
        self.base_dims = base_dims
        self.base_heads = base_heads
        self.stage_layers = stage_layers
        self.stage_count = stage_count
        self.dropout = dropout

        self.src_mask = torch.zeros((1024, 1024))  # .to('cuda')
        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.base_dims)
        if start_conv:
            module = ResidualConvBlock(self.model_dim, ConvBlock(self.model_dim, causal=False))
            self.start_conv = ConvStack(module, start_conv)
        else:
            self.start_conv = None

        self.backbone = TransformerBackbone(base_dims=base_dims, base_heads=base_heads, stage_layers=stage_layers, stage_count=4, stage_subsampling=stage_subsampling, dropout=dropout)
        self.neck = TransformerNeck(base_dims=base_dims, base_heads=base_heads, stage_layers=stage_layers, stage_count=4, stage_subsampling=stage_subsampling, dropout=dropout)

        if end_conv:
            module = ResidualConvBlock(self.model_dim, ConvBlock(self.model_dim, causal=False))
            self.end_conv = ConvStack(module, end_conv)
        else:
            self.end_conv = None

        self.decoder = torch.nn.Linear(
            in_features=self.base_dims,
            out_features=self.token_count,
            bias=True)

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        emb = self.start_conv(emb) if self.start_conv else emb
        features = self.backbone(emb)
        #for i, feature in enumerate(features):
        #    print(i, feature.shape)
        output = self.neck(features)
        output = self.end_conv(output) if self.end_conv else output
        logits = self.decoder(output)
        return logits


def main():
    token_count = 256
    model = TextTransformerMultiscale(
        token_count=token_count,
        base_dims=64, base_heads=2, stage_layers=2, stage_count=4, stage_subsampling=4, start_conv=0, end_conv=0, dropout=0.025)
    print(model)
    x = torch.randint(0, token_count, (2, 256))
    y = model(x)
    print(y.shape, x.shape)


if __name__ == '__main__':
    main()

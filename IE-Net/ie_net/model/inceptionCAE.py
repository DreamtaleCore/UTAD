"""
A modified version mvtecCAE
Model inspired by: https://github.com/natasasdj/anomalyDetection
"""

import torch
import torch.nn as nn
import torchvision.models.resnet


class Model(nn.Module):
    def __init__(self, color_mode='rgb', input_size=256, pad_type='reflect', for_info=False, n_filters=128):
        super().__init__()
        self.channels   = 1 if color_mode == 'grayscale' else 3
        assert input_size in [256, 128, 64]
        self.input_size = input_size
        self.rescale    = 1. / 255
        self.v_min      = 0.0
        self.v_max      = 1.0
        self.filters              = [n_filters//4, n_filters//2, n_filters]
        self.local_feature_shape  = (self.filters[2]*4, 4, 4)
        self.global_feature_shape = (self.filters[2],   1, 1)
        self.pad_type             = pad_type
        self.for_info             = for_info

        if self.input_size == 256:
            # Encode -------------------------------------------------------------------------- input 256x256
            self.encoder_f  = nn.Sequential(
                InceptionBlock(self.channels,   self.filters[0]),
                nn.MaxPool2d((4, 4), (4, 4)),                                                       # 64 x 64
                InceptionBlock(self.filters[0]*4, self.filters[0]),
                nn.MaxPool2d((2, 2), (2, 2)),                                                       # 32 x 32
                InceptionBlock(self.filters[0]*4, self.filters[1]),
                nn.MaxPool2d((2, 2), (2, 2)),                                                       # 16 x 16
                InceptionBlock(self.filters[1]*4, self.filters[2]),
                nn.MaxPool2d((2, 2), (2, 2)),                                                        #  8 x  8
            )

            # Decode -------------------------------------------------------------------------- input  8 x  8
            self.decoder = nn.Sequential(
                InceptionBlock(self.filters[2]*4, self.filters[2]),
                nn.Upsample(scale_factor=2),                                                        # 16 x 16
                InceptionBlock(self.filters[2]*4, self.filters[1]),
                nn.Upsample(scale_factor=4),                                                        # 64 x 64
                InceptionBlock(self.filters[1]*4, self.filters[1]),
                nn.Upsample(scale_factor=2),                                                        # 128x128
                InceptionBlock(self.filters[1]*4, self.filters[0]),
                nn.Upsample(scale_factor=2),                                                        # 256x256
                nn.Conv2d(self.filters[0]*4, self.channels, (1, 1), stride=1, bias=False),
                nn.Sigmoid(),
            )
        
        # Info mapping ------------------------------------------------------------------------ input  4 x  4
        self.encoder_g = nn.Sequential(
            nn.Conv2d(self.filters[2]*4,          64, (3, 3), stride=1, padding=1), nn.ReLU(),      #  4 x  4
            nn.Conv2d(64,          self.filters[2]*4, (3, 3), stride=1, padding=1),                 #  4 x  4
            nn.AvgPool2d((4, 4))
        )
        self.down_2x2 = nn.MaxPool2d((2, 2), (2, 2))
        self.encoder_mean    = nn.Linear(self.filters[2]*4, self.filters[2])
        self.encoder_log_var = nn.Linear(self.filters[2]*4, self.filters[2])
    
    def forward(self, input):
        latent = self.encoder_f(input)
        recons = self.decoder(latent)
        
        latent = self.down_2x2(latent)
        latent_info = self.encoder_g(latent)
        mean        = self.encoder_mean(latent_info.view(latent_info.shape[0], -1))
        log_var     = self.encoder_log_var(latent_info.view(latent_info.shape[0], -1))

        if not self.for_info:
            mean.detach()
            log_var.detach()
        
        return recons, latent, mean, log_var
        # return recons, None, None, None

    
class InceptionBlock(nn.Module):
    """
        1x1 convolution
        x0 = Conv2D(
            filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        x0 = BatchNormalization()(x0)
        x0 = LeakyReLU(alpha=0.1)(x0)
        # 3x3 convolution
        x1 = Conv2D(
            filters, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        # 5x5 convolution
        x2 = Conv2D(
            filters, (5, 5), padding="same", kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        x2 = BatchNormalization()(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)
        # Max Pooling
        x3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
        x3 = Conv2D(
            filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
        )(x3)
        x3 = BatchNormalization()(x3)
        x3 = LeakyReLU(alpha=0.1)(x3)
        output = concatenate([x0, x1, x2, x3], axis=3)
    """
    def __init__(self, in_dim, filters):
        super().__init__()
        self.block1x1 = nn.Sequential(
            nn.Conv2d(in_dim, filters, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters, eps=0.001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.block3x3 = nn.Sequential(
            nn.Conv2d(in_dim, filters, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters, eps=0.001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.block5x5 = nn.Sequential(
            nn.Conv2d(in_dim, filters, (5, 5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(filters, eps=0.001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        self.block_mp = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=1, padding=1),
            nn.Conv2d(in_dim, filters, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters, eps=0.001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    
    def forward(self, x):
        x0 = self.block1x1(x)
        x1 = self.block3x3(x)
        x2 = self.block5x5(x)
        x3 = self.block_mp(x)

        output = torch.cat([x0, x1, x2, x3], dim=1)
        return output


if __name__ == "__main__":
    rand_in = torch.randn((1, 3, 256, 256)).float()
    model = Model()
    # a = model.encoder_f(rand_in)
    a, b, c, d = model(rand_in)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)



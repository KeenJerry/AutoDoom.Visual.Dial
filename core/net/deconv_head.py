from torch import nn


class DeconvHead(nn.Module):
    def __init__(self, output_channels):
        super(DeconvHead, self).__init__()

        self.layer1_conv_transpose_2d = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4),
                                                           stride=(2, 2), padding=(1, 1), output_padding=(0, 0),
                                                           bias=False)
        self.layer1_batch_norm_2d = nn.BatchNorm2d(256)
        self.layer1_relu1 = nn.ReLU(inplace=True)

        self.layer2_conv_transpose_2d = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4, 4),
                                                           stride=(2, 2), padding=(1, 1), output_padding=(0, 0),
                                                           bias=False)
        self.layer2_batch_norm_2d = nn.BatchNorm2d(256)
        self.layer2_relu1 = nn.ReLU(inplace=True)

        self.layer3_conv_transpose_2d = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4, 4),
                                                           stride=(2, 2), padding=(1, 1), output_padding=(0, 0),
                                                           bias=False)
        self.layer3_batch_norm_2d = nn.BatchNorm2d(256)
        self.layer3_relu1 = nn.ReLU(inplace=True)

        self.layer4_conv_2d = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=(1, 1),
                                        padding=(0, 0), bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        x = self.layer1_conv_transpose_2d(x)
        x = self.layer1_batch_norm_2d(x)
        x = self.layer1_relu1(x)

        x = self.layer2_conv_transpose_2d(x)
        x = self.layer2_batch_norm_2d(x)
        x = self.layer2_relu1(x)

        x = self.layer3_conv_transpose_2d(x)
        x = self.layer3_batch_norm_2d(x)
        x = self.layer3_relu1(x)

        x = self.layer4_conv_2d(x)

        return x

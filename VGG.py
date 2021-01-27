from torch import nn
import math

def_full_layer = [2048,4096,4096]

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True, full_layer=None):
        super(VGG, self).__init__()
        self.features = features
        if full_layer is None:
            full_layer = def_full_layer
            
        self.classifier = nn.Sequential(
            nn.Linear(full_layer[0], full_layer[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(full_layer[1], full_layer[2]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(full_layer[2], num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'AE1': [64, 'M', 128, 'M', 256, 128],
    'AE2': [64, 'M', 128, 'M', 256, 256, 'M', 512, 256],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


def vgg11():
    model = VGG(make_layers(cfg['A']))
    return model

def vgg19_bn():
    model = VGG(make_layers(cfg['E'], batch_norm=True))
    return model

def vgg11E1():
    model = VGG(make_layers(cfg['AE1']),full_layer=[8192,1024,1024])
    return model

def vgg11E2():
    model = VGG(make_layers(cfg['AE2']),full_layer=[4096,2048,2048])
    return model

if __name__ == "__main__":
    vgg = vgg11E1()
    print(vgg)

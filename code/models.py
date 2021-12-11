
import models.densenet3D as DN3
import models.resnet3D as Res3

def generate_model(model_depth, **kwargs):
    assert model_depth in ['Densenet121', 'Densenet169', 'Densenet201', 'Densenet264','Resnet10', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Resnet200']

    if model_depth == 'Densenet121':
        model = DN3.DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 'Densenet169':
        model = DN3.DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 'Densenet201':
        model = DN3.DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 'Densenet264':
        model = DN3.DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    elif model_depth == 'Resnet10':
        model = Res3.ResNet(Res3.BasicBlock, [1, 1, 1, 1], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet18':
        model = Res3.ResNet(Res3.BasicBlock, [2, 2, 2, 2], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet34':
        model = Res3.ResNet(Res3.BasicBlock, [3, 4, 6, 3], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet50':
        model = Res3.ResNet(Res3.Bottleneck, [3, 4, 6, 3], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet101':
        model = Res3.ResNet(Res3.Bottleneck, [3, 4, 23, 3], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet152':
        model = Res3.ResNet(Res3.Bottleneck, [3, 8, 36, 3], Res3.get_inplanes(), **kwargs)
    elif model_depth == 'Resnet200':
        model = Res3.ResNet(Res3.Bottleneck, [3, 24, 36, 3], Res3.get_inplanes(), **kwargs)

    return model

import functools

# import your model builders here
# from models.resnet import resnet50, revnet50
# from models.vggnet import vgg19
from models.pytorch_resnet import resnet50

def get_net(cfg, num_classes=None):
    """
    PyTorch equivalent of the TensorFlow helper.

    Args:
        cfg: config object with attributes like:
            - architecture
            - filters_factor
            - last_relu
            - mode
            - task
            - weight_decay
        num_classes: output classes for classifier head

    Returns:
        A partially-configured model constructor.
        Call it with no arguments to build the model:
            model_fn = get_net(cfg, num_classes=1000)
            model = model_fn()
    """
    architecture = cfg.architecture

    # if "vgg19" in architecture:
    #     net = functools.partial(
    #         vgg19,
    #         filters_factor=getattr(cfg, "filters_factor", 8),
    #     )
    # else:
    if "resnet50" in architecture:
        net = resnet50
    # elif "revnet50" in architecture:
    #     net = revnet50
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    net = functools.partial(
        net,
        filters_factor=getattr(cfg, "filters_factor", 4),
        last_relu=getattr(cfg, "last_relu", True),
        mode=getattr(cfg, "mode", "v2"),
    )

    if getattr(cfg, "task", None) in ("jigsaw", "relative_patch_location"):
        net = functools.partial(
            net,
            root_conv_stride=1,
            strides=(2, 2, 1),
        )

    # Common settings across all models
    net = functools.partial(
        net,
        num_classes=num_classes,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    return net
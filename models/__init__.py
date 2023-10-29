from .squeezenet import SqueezeNet
from .mobilenet import MobileNet
from .xceptionnet import XceptionNet
from .shufflenet import ShuffleNet

squeezenet_model = SqueezeNet(num_classes=2)
mobilenet_model = MobileNet(num_classes=2)
xceptionnet_model = XceptionNet(num_classes=2)
shufflenet_model = ShuffleNet(num_classes=2)
# 允许调用四种模型

from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(name, pretrained = True):
    model = eval(name)(pretrained = pretrained)
    return model
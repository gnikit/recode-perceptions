import torchvision.models as models
import torch
import torch.nn as nn

def initialise_model():
    model = models.resnet101(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-2]))
    return model

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.pretrained = initialise_model()
        self.my_new_layers = nn.Sequential(nn.Flatten(),
                                            nn.Linear(2048*7*7, 512),
                                            nn.Linear(512, 256),
                                            nn.Linear(256,1))

    def forward(self, x):
        x = x.float()
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return 
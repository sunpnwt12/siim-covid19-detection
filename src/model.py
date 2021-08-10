from timm.models.efficientnet import tf_efficientnet_b7_ns
from lib.include import *

class Effnet(nn.Module):
    def __init__(self, model_name=tf_efficientnet_b7_ns, num_classes=4, pretrained=True):
        super().__init__()
        eff = model_name(pretrained=pretrained, drop_rate=0.3, drop_path_rate=0.2, in_chans=1)
        self.b0 = nn.Sequential(
            eff.conv_stem,
            eff.bn1,
            eff.act1,
        )
        
        self.b1 = eff.blocks[0]
        self.b2 = eff.blocks[1]
        self.b3 = eff.blocks[2]
        self.b4 = eff.blocks[3]
        self.b5 = eff.blocks[4]
        self.b6 = eff.blocks[5]
        self.b7 = eff.blocks[6]
        self.b8 = nn.Sequential(
            eff.conv_head,
            eff.bn2,
            eff.act2,
        )
        self.logit = nn.Linear(2560, num_classes) # effnetb7_ns
    
        self.mask = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1), # effnetb7_ns
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = 2 * x - 1
        batch_size = len(x)
        
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        #------------
        aux = self.mask(x)
        #-------------
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        #x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, aux


def convert_silu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, child_name, nn.Mish(inplace=True))
        else:
            convert_silu_to_mish(child)
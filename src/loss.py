from lib.include import *
from lib.lovasz_losses import lovasz_hinge

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs) # fixed for mixed precision
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean') # fixed for mixed precision
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class LovaszHinge(nn.Module):
    def __init__(self):
        super(LovaszHinge, self).__init__()

    def forward(self, inputs, targets):
        loss = lovasz_hinge(inputs, targets)
        return loss
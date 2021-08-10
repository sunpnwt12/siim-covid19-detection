from lib.include import *
from CONFIG import GlobalConfig

cfg = GlobalConfig()
class Metrics:
    def __init__(self, device):
        self.device = device
        self._reset()
        

    def _reset(self):
        self.accuracu_list = []
        self.proba_list = []
        self.truth_list = []
        
        self.accuracy = 0
        self.mAP = 0
        self.AUROC = 0
        
        self.acc = Accuracy().to(self.device)
        self.AP = AveragePrecision(num_classes=cfg.num_classes).to(self.device)
        self.auroc = AUROC(num_classes=cfg.num_classes).to(self.device)
        
    def update(self, y_onehot, proba, pred_onehot):
        self.accuracu_list.append(self.acc(pred_onehot, y_onehot).item())
        self.accuracy = torch.as_tensor(self.accuracu_list).mean()
        
        self.proba_list.append(proba)
        self.truth_list.append(y_onehot)
        
    def calculate(self):
        self.proba = torch.cat(self.proba_list)
        self.truth = torch.cat(self.truth_list)
        self.mAP = torch.stack(self.AP(self.proba, self.truth)).mean().item()
        self.AUROC = self.auroc(self.proba, self.truth).item()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

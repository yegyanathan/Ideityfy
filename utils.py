import torch
import platform


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __call__(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def getPlatform():
    plt = platform.system()
    if plt=='Darwin':
        return 'mac'
    return plt



def hasGPU(plt:str):
    if plt == 'mac':
        return torch.backends.mps.is_available()
    return torch.cuda.is_available()
    


def getDevice(plt:str):
    if plt == 'mac':
        return torch.device('mps')
    return torch.device('cuda')



def disableWarnings():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
    warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.ppo_config")
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")


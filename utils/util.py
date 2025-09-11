import argparse
import random
import numpy as np
import torch
import logging
import torch.optim as optim
import os

# logging system
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log(message="", object=None, type=logging.INFO):
    if type == logging.INFO:
        logger.info(f"\n{message}: {object}")
    elif type == logging.DEBUG:
        logger.debug(f"\n{message}: {object}")
    elif type == logging.ERROR:
        logger.error(f"\n{message}: {object}")
        
# for parsing boolean data
def str2bool(v):
    """Convert string to boolean value."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# init the seed
def init_seed(seed=1):
    os.environ["PYTHONHASHSEED"] = str(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for the data loader to load random data with random augmentation value
def worker_init_fn(worker_id):
    """Initialize random seed for DataLoader worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# warmup for epoch
class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch - 1:
            if self.after_scheduler is not None:
                return self.after_scheduler.get_lr()
            return [base_lr for base_lr in self.base_lrs]
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if self.after_scheduler is not None:
                self.after_scheduler.step()
            return
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch >= self.total_epoch - 1:
            self.finished = True
            if self.after_scheduler is not None:
                self.after_scheduler.step()
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            

import torch.optim

from common.services.trainning_config_service import LR


def get_optimizer(net):
    return torch.optim.Adam(net.parameters(), lr=LR)


def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(i) for i in [90, 110]], gamma=0.1)
    return scheduler

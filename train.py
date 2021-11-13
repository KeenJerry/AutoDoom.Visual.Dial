from torch.utils.data import DataLoader

from common.services.dataset_config_service import DATASET_TYPE
from common.services.trainning_config_service import TOTAL_EPOCH
from common.tools.parallels import DataParallelModel, DataParallelCriterion
from core.data.dataset import DialButtonDataset
from core.loss.dial_loss import DialLoss
from core.loss.optimizer import get_optimizer, get_scheduler
from core.net.dialnet import DialNet
from core.process import train_dialnet


def _main() -> None:
    # load data from file
    train_dataset = DialButtonDataset(DATASET_TYPE.train)
    test_dataset = DialButtonDataset(DATASET_TYPE.test)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=10)

    # load network
    dial_net = DialNet()
    dial_net.init_resnet_backbone()
    dial_net = DataParallelModel(dial_net).cuda()

    # get optimizer and scheduler
    optimizer = get_optimizer(dial_net)
    scheduler = get_scheduler(optimizer)

    # load loss function
    dial_loss_func = DataParallelCriterion(DialLoss())

    # run train loop on epoch
    for epoch in range(TOTAL_EPOCH):
        # run train loop on batch
        train_loss = train_dialnet(dial_net, train_dataloader, optimizer, dial_loss_func)

    # run updated network on test dataset


if __name__ == '__main__':
    _main()

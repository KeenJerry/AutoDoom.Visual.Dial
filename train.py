import logging
import os.path
import time

import torch
from torch.utils.data import DataLoader

from common.services.dataset_config_service import DATASET_TYPE
from common.services.model_storage_service import get_model_save_path
from common.services.trainning_config_service import TOTAL_EPOCH, LR
from common.tools.parallels import DataParallelModel, DataParallelCriterion
from core.data.dataset import DialButtonDataset
from core.loss.dial_loss import DialLoss
from core.loss.optimizer import get_optimizer, get_scheduler
from core.net.dialnet import DialNet
from core.net.model_state import ModelState
from core.process import train_dialnet, test_dialnet


def _main() -> None:
    # load data from file
    logging.info("LOADING TRAIN & TEST DATASET...")
    train_dataset = DialButtonDataset(DATASET_TYPE.train)
    test_dataset = DialButtonDataset(DATASET_TYPE.test)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    logging.info("Done. \t TRAIN_SIZE: {}, TEST_SIZE: {}, BATCH_SIZE: {}, WORKERS: {}".format(
        len(train_dataset.dial_keyboards), len(test_dataset.dial_keyboards), train_dataloader.batch_size,
        train_dataloader.num_workers)
    )

    # load network
    logging.info("LOADING DIAL_NETWORK...")
    dial_net = DialNet()
    dial_net.init_resnet_backbone()
    dial_net = DataParallelModel(dial_net).cuda()
    logging.info("Done")

    # get optimizer and scheduler
    optimizer = get_optimizer(dial_net)
    scheduler = get_scheduler(optimizer)

    # load loss function
    dial_loss_func = DataParallelCriterion(DialLoss())

    # check cache (latest_model)
    latest_model_path = os.path.join(get_model_save_path(), "latest_model.pth")
    if os.path.exists(latest_model_path):
        network_model_state = torch.load(latest_model_path)
        dial_net.load_state_dict(network_model_state.network_state_dict)
        optimizer.load_state_dict(network_model_state.optimizer_state_dict)
        scheduler.load_state_dict(network_model_state.scheduler_state_dict)
    else:
        network_model_state = ModelState()

    # log info
    logging.info("TOTAL_NETWORK_PARAMETERS: {:.2f}M".format(sum(p.numel() for p in dial_net.parameters()) / 1000000.0))
    logging.info("TOTAL_EPOCH: {}, LEARNING_RATE: {}".format(TOTAL_EPOCH, LR))
    logging.info("TRAIN_DATASET_SIZE: {}, TEST_DATASET_SIZE: {}".format(len(train_dataset.dial_keyboards),
                                                                        len(test_dataset.dial_keyboards)))

    logging.info("START TRAINING FROM EPOCH NUMBER: {}".format(network_model_state.epoch_number + 1))
    # run train loop on epoch
    for epoch in range(network_model_state.epoch_number + 1, TOTAL_EPOCH):
        logging.info("EPOCH NUMBER: {} START...".format(epoch))

        # run train loop on batch
        train_begin_time = time.time()
        train_loss = train_dialnet(dial_net, train_dataloader, optimizer, dial_loss_func)
        train_end_time = time.time()

        # run updated network on test dataset
        test_begin_time = time.time()
        test_loss = test_dialnet(dial_net, test_dataloader, dial_loss_func)
        test_end_time = time.time()

        # update scheduler
        scheduler.step()

        # update network model state and save
        network_model_state.update(epoch, dial_net.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                                   train_loss, test_loss)
        network_model_state.save()

        logging.info(
            "EPOCH NUMBER: {} END, train_time: {:.2f}s, test_time: {:.2f}s, train_loss: {:.2f}, test_loss: {:.2f},"
            " LR: {}".format(epoch, train_end_time - train_begin_time, test_end_time - test_begin_time, train_loss,
                             test_loss, scheduler.get_last_lr())
        )


if __name__ == '__main__':
    # set logger basic
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    _main()

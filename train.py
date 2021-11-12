from torch.utils.data import DataLoader

from common.services.dataset_config_service import DATASET_TYPE
from core.data.dataset import DialButtonDataset


def _train() -> None:
    # load data from file
    train_dataset = DialButtonDataset(DATASET_TYPE.train)
    test_dataset = DialButtonDataset(DATASET_TYPE.test)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=10)

    # load network


    # load loss function

    # run train loop on epoch
        # run train loop on batch
            # clean gradient
            # calculate loss
            # do backward propagation
            # apply weight change

    # run updated network on test dataset


if __name__ == '__main__':
    _train()

from common.services.dataset_config_service import DATASET_TYPE
from core.data.dataset import DialButtonDataset


def _train() -> None:
    # load data from file
    train_data_set = DialButtonDataset(DATASET_TYPE.train)
    test_data_set = DialButtonDataset(DATASET_TYPE.test)

    # load loss function

    # load network

    # run train loop on epoch
        # run train loop on batch
            # clean gradient
            # calculate loss
            # do backward propagation
            # apply weight change

    # run updated network on test dataset


if __name__ == '__main__':
    _train()

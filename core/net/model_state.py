import os.path

import torch

from common.services.model_storage_service import get_model_save_path


class ModelState:
    def __init__(self):
        self.epoch_number = None
        self.network_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.train_loss = None
        self.test_loss = None

    def update(self, epoch_number, network_state_dict, optimizer_state_dict, scheduler_state_dict, train_loss,
               test_loss):
        self.epoch_number = epoch_number
        self.network_state_dict = network_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.scheduler_state_dict = scheduler_state_dict
        self.train_loss = train_loss
        self.test_loss = test_loss

    def save(self):
        model_name = "{}_epoch.pth".format(self.epoch_number)
        path = get_model_save_path()
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self, os.path.join(path, model_name))

        torch.save(self, os.path.join(path, "latest_model.pth"))

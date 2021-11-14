import torch


class LossRecorder:
    def __init__(self):
        self.sum: torch.Tensor = torch.zeros(1).cuda()
        self.item_count = 0

    def update_loss(self, avg_loss, batch_size):
        self.sum += avg_loss * batch_size
        self.item_count += batch_size

    def get_avg_loss(self):
        return (self.sum / self.item_count).item()


loss_recorder = LossRecorder()

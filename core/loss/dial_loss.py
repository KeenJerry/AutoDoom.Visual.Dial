from torch import nn

from core.loss.heatmap_loss import heatmap_loss_func
from core.loss.pull_push_loss import pull_push_loss_func


def _assert_no_gradient(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class DialLoss(nn.Module):
    def __init__(self):
        super(DialLoss, self).__init__()
        self.heatmap_loss = heatmap_loss_func
        self.pull_push_loss = pull_push_loss_func

    def forward(self, predict_heatmap, predict_tagmap, ground_truth_heatmap, ground_truth_points):
        _assert_no_gradient(ground_truth_heatmap)
        _assert_no_gradient(ground_truth_points)

        batch_size, point_type_number, _, _ = ground_truth_heatmap.shape
        _, max_window_number, _, _ = ground_truth_points.shape

        reshaped_predict_tag_map = predict_tagmap.reshape(batch_size, -1)

        # calculate heatmap_loss
        heatmap_loss = self.heatmap_loss(predict_heatmap, ground_truth_heatmap)
        print("heat_map_loss: {:.2f}".format(heatmap_loss))

        # calculate pull_push_loss
        pull_push_loss = pull_push_loss_func(predict_tagmap, ground_truth_points, expected_distance=12)

        return heatmap_loss + pull_push_loss





import logging
import time

from common.services.trainning_loss_service import loss_recorder


def train_dialnet(dial_net, train_dataloader, optimizer, loss_function):
    dial_net.train()
    for index, preprocessed_data in enumerate(train_dataloader):
        batch_begin_time = time.time()
        # clean gradient
        optimizer.zero_grad()

        # predict heatmap and tagmap
        tensor_like_image = preprocessed_data[0].cuda()
        heatmap_label = preprocessed_data[1].cuda()
        ground_truth_points = preprocessed_data[2]

        predict_heatmap, predict_tagmap = dial_net(tensor_like_image)
        del tensor_like_image

        # calculate loss, avg loss in one batch
        loss = loss_function(predict_heatmap, predict_tagmap, heatmap_label, ground_truth_points)
        del heatmap_label, ground_truth_points, predict_heatmap, predict_tagmap

        # do backward propagation
        loss.backward()

        # apply weight change
        optimizer.step()

        # recorde loss
        loss_recorder.update_loss(loss.detach(), train_dataloader.batch_size)
        # logging
        batch_end_time = time.time()
        logging.info("\tBATCH NUMBER: {}, time: {:.2f}, loss: {:.2f}".format(index, batch_end_time - batch_begin_time,
                                                                             loss.item()))

        del loss

    return loss_recorder.get_avg_loss()

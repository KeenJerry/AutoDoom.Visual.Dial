import logging
import time

import torch

from common.services.trainning_loss_service import train_loss_recorder, test_loss_recorder


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

        net_begin_time = time.time()
        predict_heatmap, predict_tagmap = dial_net(tensor_like_image)
        net_end_time = time.time()
        del tensor_like_image

        # calculate loss, avg loss in one batch
        loss = loss_function(predict_heatmap, predict_tagmap, heatmap_label, ground_truth_points)
        del heatmap_label, ground_truth_points, predict_heatmap, predict_tagmap

        # do backward propagation
        loss.backward()

        # apply weight change
        optimizer.step()

        # recorde loss
        train_loss_recorder.update_loss(loss.detach(), train_dataloader.batch_size)
        # logging
        batch_end_time = time.time()
        logging.info("\tBATCH NUMBER: {}, net_calculate_time: {:.2f}s, total_time: {:.2f}s, loss: {:.2f}".
                     format(index, net_end_time - net_begin_time, batch_end_time - batch_begin_time, loss.item()))

        del loss

    return train_loss_recorder.get_avg_loss()


def test_dialnet(dial_net, test_dataloader, loss_function):
    heat_maps = []
    tag_maps = []

    dial_net.eval()

    with torch.no_grad():
        for index, preprocessed_data in enumerate(test_dataloader):
            test_loss_recorder.reset()

            tensor_like_image = preprocessed_data[0].cuda()
            heatmap_label = preprocessed_data[1].cuda()
            ground_truth_points = preprocessed_data[2]

            predict_heatmap, predict_tagmap = dial_net(tensor_like_image)
            del tensor_like_image

            # calculate loss, avg loss in one batch
            loss = loss_function(predict_heatmap, predict_tagmap, heatmap_label, ground_truth_points)
            del heatmap_label, ground_truth_points, predict_heatmap, predict_tagmap

            test_loss_recorder.update_loss(loss.detach(), test_dataloader.batch_size)
            del loss

        return test_loss_recorder.get_avg_loss()

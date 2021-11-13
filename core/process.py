from common.services.trainning_loss_service import loss_recorder


def train_dialnet(dial_net, train_dataloader, optimizer, loss_function):
    dial_net.train()
    for index, preprocessed_data in enumerate(train_dataloader):
        # clean gradient
        optimizer.zero_grad()

        # predict heatmap and tagmap
        tensor_like_image = preprocessed_data[0].cuda()
        heatmap_label = preprocessed_data[1].cuda()
        ground_truth_points = preprocessed_data[2]

        predict_heatmap, predict_tagmap = dial_net(tensor_like_image)
        del tensor_like_image

        # calculate loss
        loss = loss_function(predict_heatmap, predict_tagmap, heatmap_label, ground_truth_points)
        del heatmap_label, ground_truth_points, predict_heatmap, predict_tagmap

        # do backward propagation
        loss.backward()

        # apply weight change
        optimizer.step()

        # recorde loss
        loss_recorder.update(loss.detach(), train_dataloader.batch_size)
        del loss

    return loss_recorder.get_avg_loss()

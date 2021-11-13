import torch


def pull_push_loss_func(predict_tagmap, ground_truth_points, expected_distance):
    batch_size, max_dial_button_number, point_number, _ = ground_truth_points.shape
    epsilon = 1e-6

    pull_losses = []
    push_losses = []

    for i in range(batch_size):
        # get all part of ground_truth_points
        point_locations_on_tagmap = ground_truth_points[i, :, :, 0].reshape(-1)
        point_visibilities = ground_truth_points[i, :, : 1]
        button_positions_on_image = ground_truth_points[i][:, :, 2].float().mean(dim=1)

        # get tag value of all points
        all_point_tag_values = predict_tagmap[i, point_locations_on_tagmap].reshape(max_dial_button_number,
                                                                                    point_number - 1, 1)

        # prepare universal parameters for push and pull losses
        button_center_tag_values = []
        button_positions = []

        max_button_position = float("-inf")
        min_button_position = float("inf")

        temp_push_loss = 0
        temp_pull_loss = 0

        # calculate pull loss
        for point_tag_values, point_visibility, button_position in zip(all_point_tag_values, point_visibilities,
                                                                       button_positions_on_image):
            valid_point_number = point_visibility.sum()
            if valid_point_number > 0:
                button_center_tag_value = point_tag_values[: valid_point_number].mean(dim=0)
                temp_pull_loss += torch.sum(torch.pow(point_tag_values[:valid_point_number] - button_center_tag_value,
                                                      2))

                # append center tag
                button_center_tag_values.append(button_center_tag_value)

                # append button position
                button_positions.append(button_position)

                # calculate min and max position
                if button_position > max_button_position:
                    max_point_position = button_position
                if button_position < min_button_position:
                    min_point_position = button_position
            else:
                break

        # prepare parameters for push loss
        max_distance_between_buttons = max_button_position - min_button_position
        valid_button_number = len(button_center_tag_values)

        push_weight = 3

        # calculate push loss
        for row in range(valid_button_number):
            for column in range(valid_button_number):
                if row != column:
                    weight = abs(button_positions[row] - button_positions[column]) / max_distance_between_buttons * \
                             push_weight + 1.0
                    tag_distance = expected_distance - torch.abs(button_center_tag_values[row] -
                                                                 button_center_tag_values[column])
                    tag_distance = weight * max(0, tag_distance)

                    temp_push_loss += tag_distance

        temp_pull_loss /= valid_button_number + epsilon
        temp_push_loss /= valid_button_number * (valid_button_number - 1) + epsilon

        pull_losses.append(temp_pull_loss)
        push_losses.append(temp_push_loss)

    return sum(push_losses) + sum(pull_losses) / batch_size

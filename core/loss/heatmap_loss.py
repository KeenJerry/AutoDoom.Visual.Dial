def heatmap_loss_func(predict_heatmap, ground_truth_heatmap):
    diff = predict_heatmap - ground_truth_heatmap
    return diff.sum() / len(predict_heatmap)

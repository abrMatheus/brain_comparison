import torch
from medpy.metric import binary


def _custom_labels(logits, targets, class_index):

    if (class_index == 1):
        # whole tumor WT = all labels
        pred, t = logits != 0, targets != 0
    elif (class_index == 2):
        # Tumor Core = label 3 + label 1
        pred, t = logits == 1 , targets == 1
        p3, t3 = logits == 3, targets == 3
        pred += p3
        t += t3
    else:
        #background or enhancing tumor
        pred, t = logits == class_index, (targets == class_index)
    return pred, t


def _apply_multiclass(logits, targets, metric):
    batch_size, class_cnt = logits.shape[0], logits.shape[1]
    logits = logits.argmax(axis=1)

    scores_list = []
    for class_index in range(class_cnt):
        predict, target = _custom_labels(logits, targets, class_index)

        predict = predict.view(batch_size, -1).cpu()
        target = target.view(batch_size, -1).cpu()

        # compute metric per sample in batch
        score = []
        for p, t in zip(predict, target):
            score.append(metric(p.numpy(), t.numpy()))

        score = torch.tensor(score).mean()
        scores_list.append(score)

    return scores_list


def _apply_binary(prediction, targets, metric):
    targets = targets != 0
    prediction = prediction.cpu()
    targets = targets.cpu()
    score = []
    for p, t in zip(prediction, targets):
        score.append(metric(p.numpy(), t.numpy()))
    return torch.tensor(score).mean()


def multiclass_dice(logits, targets):
    return _apply_multiclass(logits, targets, binary.dc)


def multiclass_hausdorff95(logits, targets):
    return _apply_multiclass(logits, targets, binary.hd95)


def multiclass_precision(logits, targets):
    return _apply_multiclass(logits, targets, binary.precision)


def multiclass_sensitivity(logits, targets):
    return _apply_multiclass(logits, targets, binary.sensitivity)


def binary_dice(prediction, targets):
    return _apply_binary(prediction, targets, binary.dc)


def binary_precision(prediction, targets):
    return _apply_binary(prediction, targets, binary.precision)


def binary_sensitivity(prediction, targets):
    return _apply_binary(prediction, targets, binary.sensivity)

import numpy as np
import os
import logging
import glob
import cv2

def cal_global_acc(pred, gt):
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls=2):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))

    tp_all = [0] * num_cls
    fp_all = [0] * num_cls
    fn_all = [0] * num_cls

    for stats in statistics:
        for i in range(num_cls):
            tp_all[i] += stats[i][0]
            fp_all[i] += stats[i][1]
            fn_all[i] += stats[i][2]

    # calculate metrics
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_cls):
        tp = tp_all[i]
        fp = fp_all[i]
        fn = fn_all[i]

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_list.append(iou)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_list.append(precision)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_list.append(recall)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)

    mIoU = np.mean(iou_list)
    # We are interested in the metrics for the 'crack' class, which is class 1
    crack_f1 = f1_list[1]
    crack_precision = precision_list[1]
    crack_recall = recall_list[1]

    return mIoU, crack_f1, crack_precision, crack_recall

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    print(f"Found {len(gt_list)} ground truth images and {len(pred_list)} predicted images.")
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

def eval(log_eval, results_dir, epoch):
    suffix_gt = "lab"
    suffix_pred = "pre"
    log_eval.info(results_dir)
    log_eval.info("checkpoints -> " + results_dir)
    src_img_list, tgt_img_list, _, _ = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    mIoU, f1, precision, recall = segment_metrics(src_img_list, tgt_img_list)

    log_eval.info("mIoU -> " + str(mIoU))
    log_eval.info("F1 -> " + str(f1))
    log_eval.info("Precision -> " + str(precision))
    log_eval.info("Recall -> " + str(recall))
    log_eval.info("eval finish!")

    return {'epoch': epoch, 'mIoU': mIoU, 'F1': f1, 'Precision': precision, 'Recall': recall}

if __name__ == '__main__':
    suffix_gt = "lab"
    suffix_pred = "pre"
    results_dir = "/results" # lab images + pre images
    src_img_list, tgt_img_list, _, _ = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    mIoU, f1, precision, recall = segment_metrics(src_img_list, tgt_img_list)

    print("mIoU -> " + str(mIoU))
    print("F1 -> " + str(f1))
    print("Precision -> " + str(precision))
    print("Recall -> " + str(recall))
    print("eval finish!")


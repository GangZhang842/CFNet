#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import time

import collections

import pdb


class PanopticEval:
  """ Panoptic evaluation using numpy
  
  authors: Andres Milioto and Jens Behley
  """

  def __init__(self, Classes, device=None, ignore=None, offset=2**32, min_points=30):
    self.Classes = Classes
    self.n_classes = len(Classes) + 1
    assert (device == None)
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

    print("[PANOPTIC EVAL] IGNORE: ", self.ignore)
    print("[PANOPTIC EVAL] INCLUDE: ", self.include)

    self.reset()
    self.offset = offset  # largest number of instances in a given scan
    self.min_points = min_points  # smallest number of points to consider instances in gt
    self.eps = 1e-15

  def num_classes(self):
    return self.n_classes

  def reset(self):
    # general things
    # iou stuff
    self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
    # panoptic stuff
    self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
    self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

  ################################# IoU STUFF ##################################
  def addBatchSemIoU(self, x_sem, y_sem):
    # idxs are labels and predictions
    idxs = np.stack([x_sem, y_sem], axis=0)

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

  def getSemIoUStats(self):
    # clone to avoid modifying the real deal
    conf = self.px_iou_conf_matrix.copy().astype(np.double)
    # remove fp from confusion on the ignore classes predictions
    # points that were predicted of another class, but were ignore
    # (corresponds to zeroing the cols of those classes, since the predictions
    # go on the rows)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diagonal()
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getSemIoU(self):
    tp, fp, fn = self.getSemIoUStats()
    # print(f"tp={tp}")
    # print(f"fp={fp}")
    # print(f"fn={fn}")
    intersection = tp
    union = tp + fp + fn
    union = np.maximum(union, self.eps)
    iou = intersection.astype(np.double) / union.astype(np.double)
    iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()

    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getSemAcc(self):
    tp, fp, fn = self.getSemIoUStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum()
    total = np.maximum(total, self.eps)
    acc_mean = total_tp.astype(np.double) / total.astype(np.double)

    return acc_mean  # returns "acc mean"

  ################################# IoU STUFF ##################################
  ##############################################################################

  #############################  Panoptic STUFF ################################
  def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
    # make sure instances are not zeros (it messes with my approach)
    x_inst_row = x_inst_row + 1
    y_inst_row = y_inst_row + 1

    # only interested in points that are outside the void area (not in excluded classes)
    for cl in self.ignore:
      # make a mask for this class
      gt_not_in_excl_mask = y_sem_row != cl
      # remove all other points
      x_sem_row = x_sem_row[gt_not_in_excl_mask]
      y_sem_row = y_sem_row[gt_not_in_excl_mask]
      x_inst_row = x_inst_row[gt_not_in_excl_mask]
      y_inst_row = y_inst_row[gt_not_in_excl_mask]

    # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
    for cl in self.include:
      # print("*"*80)
      # print("CLASS", cl.item())
      # get a class mask
      x_inst_in_cl_mask = x_sem_row == cl
      y_inst_in_cl_mask = y_sem_row == cl

      # get instance points in class (makes outside stuff 0)
      x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
      y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

      # generate the areas for each unique instance prediction
      unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
      id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
      matched_pred = np.array([False] * unique_pred.shape[0])
      # print("Unique predictions:", unique_pred)

      # generate the areas for each unique instance gt_np
      unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
      id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
      matched_gt = np.array([False] * unique_gt.shape[0])
      # print("Unique ground truth:", unique_gt)

      # generate intersection using offset
      valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
      offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
      unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

      # generate an intersection map
      # count the intersections with over 0.5 IoU as TP
      gt_labels = unique_combo // self.offset
      pred_labels = unique_combo % self.offset
      gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
      pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
      intersections = counts_combo
      unions = gt_areas + pred_areas - intersections
      ious = intersections.astype(np.float) / unions.astype(np.float)


      tp_indexes = ious > 0.5
      self.pan_tp[cl] += np.sum(tp_indexes)
      self.pan_iou[cl] += np.sum(ious[tp_indexes])

      matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
      matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

      # count the FN
      self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

      # count the FP
      self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

  def getPQ(self):
    # first calculate for all classes
    sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
    rq_all = self.pan_tp.astype(np.double) / np.maximum(
        self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
        self.eps)
    pq_all = sq_all * rq_all

    # then do the REAL mean (no ignored classes)
    SQ = sq_all[self.include].mean()
    RQ = rq_all[self.include].mean()
    PQ = pq_all[self.include].mean()

    return PQ, SQ, RQ, pq_all, sq_all, rq_all

  #############################  Panoptic STUFF ################################
  ##############################################################################

  def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    self.addBatchSemIoU(x_sem, y_sem)

    # now do the panoptic stuff
    self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)
  
  def get_metric(self):
    result_dic = collections.OrderedDict()

    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = self.getPQ()
    class_IoU, class_all_IoU = self.getSemIoU()

    # now make a nice dictionary
    output_dict = {}
    class_PQ = class_PQ.item()
    class_SQ = class_SQ.item()
    class_RQ = class_RQ.item()
    class_IoU = class_IoU.item()
    class_all_PQ = class_all_PQ.flatten().tolist()[1:]
    class_all_SQ = class_all_SQ.flatten().tolist()[1:]
    class_all_RQ = class_all_RQ.flatten().tolist()[1:]
    class_all_IoU = class_all_IoU.flatten().tolist()[1:]

    output_dict["all"] = {}
    output_dict["all"]["PQ"] = class_PQ
    output_dict["all"]["SQ"] = class_SQ
    output_dict["all"]["RQ"] = class_RQ
    output_dict["all"]["IoU"] = class_IoU
    for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
      class_str = self.Classes[idx]
      output_dict[class_str] = {}
      output_dict[class_str]["PQ"] = pq
      output_dict[class_str]["SQ"] = sq
      output_dict[class_str]["RQ"] = rq
      output_dict[class_str]["IoU"] = iou
    
    # split things and stuff
    things = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
    stuff = ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk',
    'terrain', 'pole', 'traffic-sign']
    all_classes = things + stuff

    PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
    PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
    RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
    SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])

    PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
    RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
    SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])

    PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
    RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
    SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])

    result_dic["pq_mean"] = float(PQ_all)
    result_dic["pq_dagger"] = float(PQ_dagger)
    result_dic["sq_mean"] = float(SQ_all)
    result_dic["rq_mean"] = float(RQ_all)
    result_dic["iou_mean"] = float(class_IoU)
    result_dic["pq_stuff"] = float(PQ_stuff)
    result_dic["rq_stuff"] = float(RQ_stuff)
    result_dic["sq_stuff"] = float(SQ_stuff)
    result_dic["pq_things"] = float(PQ_things)
    result_dic["rq_things"] = float(RQ_things)
    result_dic["sq_things"] = float(SQ_things)
    return result_dic
from roma import console



def leg_move_evaluation(intv_gt, intv_detec, report=True, alpha=0.5):
  """
  in - ground-truth intervals
     - detection intervals
  out - precision
      - recall
      - three error lists
  """
  TP = 0

  FN_list = []
  FP_list = []
  local_error_list = []
  num_d = len(intv_detec)
  num_gt = len(intv_gt)

  flag = 0
  intervals = sorted(intv_gt + intv_detec)
  for i, (x1, x2) in enumerate(intervals, 1):
    iou = 0
    if flag:
      flag = 0
      continue
    if i == len(intervals): (x3, x4) = (x2 + 1, x2 + 2)
    else: (x3, x4) = intervals[i]
    if x3 < x2:
      iou = (x4 - x3)/(x2 - x1) if x4 < x2  else (x2 - x3)/(x4 - x1)
      if iou > alpha: TP += 1
      else: local_error_list.append((x1, x2))
      flag = 1
    elif (x1, x2) not in local_error_list:
      if (x1, x2) in intv_gt: FN_list.append((x1, x2))
      elif (x1, x2) in intv_detec: FP_list.append((x1, x2))

  precision = TP / num_d
  recall = TP / num_gt

  return TP, precision, recall, FN_list, FP_list, local_error_list


def search_error(self, intv1, intv2):
  pass
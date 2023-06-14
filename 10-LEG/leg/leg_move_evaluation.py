from roma import console



def leg_move_evaluation(intv_gt, intv_detec, report = True):
    """
    in - gt intervals
       - detection intervals
    out - precision
        - recall
    """
    alpha = 0.5
    TP = 0

    error_list = []
    num_d = len(intv_detec)
    num_gt = len(intv_gt)

    intervals = sorted(intv_gt + intv_detec)
    for i, (x1, x2) in enumerate(intervals, 1):
        iou = 0
        if i == len(intervals):break
        (x3, x4) = intervals[i]

        if x3 < x2:
            iou = (x4 - x3)/(x2 - x1) if x4 < x2  else (x2 - x3)/(x4 - x1)

            if iou > alpha: TP += 1
            else: error_list.append((x1, x2))

    precision = TP/num_d
    recall = TP/num_gt
    if report:
        console.show_info(f'alpha evaluation result: '
                          f'precision = {precision:.3f}, recall = {recall:.3f}, '
                          f'TP = {TP}, ground_truth num = {num_gt}, detection num = {num_d},'
                          )
    return error_list

def search_error(self, intv1, intv2):
    pass
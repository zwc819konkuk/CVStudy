import numpy as np


def nms(boxes, score, thre):
    # 容错处理
    if len(boxes) == 0:
        return [], []
    # 类型转换
    # boxes使用的是极坐标表示方式
    boxes = np.array(boxes)
    score = np.array(score)
    # 获取左上角和右下角坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # 计算面积
    areas = (x2 - x1) * (y2 - y1)
    # NMS
    picked_boxes = []
    picked_scores = []
    # 排序:从小到大
    order = np.argsort(score)
    while order.size > 0:
        # 获取最大索引
        index = order[-1]
        # 最后保留下来
        picked_boxes.append(boxes[index])
        picked_scores.append(score[index])
        # IOU
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        inter_area = w * h
        iou = inter_area / (areas[index] + areas[order[:-1]] - inter_area)
        # 删除冗余框
        keep_boxes = np.where(iou < thre)
        # 更新order
        order = order[keep_boxes]
    return picked_boxes, picked_scores


bounding = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
confidence_score = [0.9, 0.65, 0.8]
threshold = 0.3
picked_boxes, picked_score = nms(bounding, confidence_score, threshold)
print('阈值threshold为:', threshold)
print('NMS后得到的bbox是：', picked_boxes)
print('NMS后得到的bbox的confidences是：', picked_score)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 定义方法计算IOU
def Iou(box1, box2, wh=False):
    # 判断bbox的表示形式
    if wh == False:
        # 使用极坐标形式表示：直接获取两个bbox的坐标
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        # 使用中心点形式表示： 获取两个两个bbox的极坐标表示形式
        # 第一个框左上角坐标
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        # 第一个框右下角坐标
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        # 第二个框左上角坐标
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        # 第二个框右下角坐标
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou
# 真实框与预测框
True_bbox, predict_bbox = [100, 35, 398, 400], [40, 150, 355, 398]
# bbox是bounding box的缩写
img = plt.imread('dog.jpeg')
fig = plt.imshow(img)
# 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：((左上x, 左上y), 宽, 高)
# 真实框绘制
fig.axes.add_patch(plt.Rectangle(
    xy=(True_bbox[0], True_bbox[1]), width=True_bbox[2]-True_bbox[0], height=True_bbox[3]-True_bbox[1],
    fill=False, edgecolor="blue", linewidth=2))
# 预测框绘制
fig.axes.add_patch(plt.Rectangle(
    xy=(predict_bbox[0], predict_bbox[1]), width=predict_bbox[2]-predict_bbox[0], height=predict_bbox[3]-predict_bbox[1],
    fill=False, edgecolor="red", linewidth=2))
plt.show()
print(Iou(True_bbox,predict_bbox))
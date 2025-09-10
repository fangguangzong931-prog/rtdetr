# 文件路径: rtdetr_pytorch/tools/my_test.py
# (已添加“打印各类别精度”功能)

import os
import sys

# 添加项目根目录到Python路径，以便导入src中的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 从项目中导入必要的模块
from src.core import YAMLConfig
from src.solver import TASKS
from src.data.coco.coco_eval import CocoEvaluator
from src.misc.dist import dist_init, device


# --- 自定义评估逻辑 ---
def calculate_and_print_custom_metrics(tp, fp, tn, fn):
    """根据TP, FP, TN, FN计算并打印性能指标"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n--- 场景分类评估完成 (存在/不存在鸟类) ---")
    print("混淆矩阵结果:")
    print(f"- 真阳性 (TP): {tp} (有鸟, 且正确检测到)")
    print(f"- 真阴性 (TN): {tn} (无鸟, 且未检测到)")
    print(f"- 假阳性 (FP): {fp} (无鸟, 但误报为有鸟)")
    print(f"- 假阴性 (FN): {fn} (有鸟, 但未能检测到)")
    print("\n性能指标:")
    print(f"- 精确率 (Precision): {precision:.4f}")
    print(f"- 召回率 (Recall):    {recall:.4f}")
    print(f"- F1 分数 (F1-Score): {f1_score:.4f}\n")


# === 新增函数：打印每个类别的AP ===
def print_per_class_ap(coco_evaluator):
    """
    从coco_evaluator中提取并打印每个类别的AP值
    """
    if 'bbox' not in coco_evaluator.coco_eval:
        return

    coco_eval = coco_evaluator.coco_eval['bbox']
    cat_ids = coco_eval.params.catIds
    cat_names = [coco_evaluator.coco_gt.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

    # 获取所有类别的AP结果 (IoU=0.50:0.95, area=all, maxDets=100)
    # coco_eval.eval['precision'] 的维度是 [T, R, K, A, M]
    # T: 10个IoU阈值
    # R: 101个recall阈值
    # K: 类别数量
    # A: 4个面积范围
    # M: 3个maxDets设置
    precision = coco_eval.eval['precision']

    results_per_category = []
    for idx, catId in enumerate(cat_ids):
        # 取出特定类别(k=idx), 所有面积(a=0), maxDets=100(m=2) 的精度
        p = precision[:, :, idx, 0, 2]
        # 计算在所有IoU阈值下的平均精度
        ap = np.mean(p[p > -1])
        results_per_category.append((cat_names[idx], f'{ap:0.4f}'))

    # 按AP值降序排序
    results_per_category.sort(key=lambda x: float(x[1]), reverse=True)

    print("--- 精度汇总 (按AP值降序排列): ---")
    for name, ap in results_per_category:
        print(f"- 类别 '{name}': AP = {ap}")
    print("-" * 30)


# ==================================


@torch.no_grad()
def main(args):
    # ... (前面的代码保持不变) ...
    cfg = YAMLConfig(args.config)
    model = TASKS[cfg.yaml_cfg['task']](cfg).model
    print(f"正在加载权重: {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['ema' if 'ema' in checkpoint else 'model'])
    model.to(device())
    model.eval()
    val_loader = cfg.eval_dataloader
    coco_gt = val_loader.dataset.coco
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox',))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [cat['name'] for cat in cats]
    colors = plt.get_cmap('tab20', len(class_names))
    tp, fp, tn, fn = 0, 0, 0, 0
    BIRD_CLASS_ID = 1
    CONF_THRESHOLD = 0.5
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"画框图片将保存在: {args.save_dir}")
    print("开始评估...")
    for samples, targets in tqdm(val_loader):
        samples = samples.to(device())
        outputs = model(samples)
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

        for i in range(len(targets)):
            # ... (这部分自定义指标统计和画框的代码保持不变) ...
            target = targets[i]
            output = outputs[i]
            gt_labels = target['labels'].cpu().numpy()
            has_bird_gt = BIRD_CLASS_ID in gt_labels
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            has_bird_pred = any(
                score > CONF_THRESHOLD and label == BIRD_CLASS_ID for score, label in zip(pred_scores, pred_labels))

            if has_bird_gt and has_bird_pred:
                tp += 1
            elif not has_bird_gt and has_bird_pred:
                fp += 1
            elif has_bird_gt and not has_bird_pred:
                fn += 1
            elif not has_bird_gt and not has_bird_pred:
                tn += 1

            if args.save_dir:
                image_id = target['image_id'].item()
                img_info = coco_gt.loadImgs(image_id)[0]
                img_path = os.path.join(cfg.val_dataset['dataset_dir'], cfg.val_dataset['image_dir'],
                                        img_info['file_name'])
                image = cv2.imread(img_path)
                for score, label_id, box in zip(pred_scores, pred_labels, output['boxes'].cpu().numpy()):
                    if score < CONF_THRESHOLD: continue
                    class_id_index = coco_gt.getCatIds().index(label_id)
                    class_name = class_names[class_id_index]
                    color_rgb = colors(class_id_index)[:3]
                    color_bgr = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                save_path = os.path.join(args.save_dir, img_info['file_name'])
                cv2.imwrite(save_path, image)

    # 8. 评估结束，汇总并打印结果
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()

    # 打印COCO的12项总览评估结果
    coco_evaluator.summarize()

    # === 新增代码：调用新函数，打印每个类别的AP ===
    print_per_class_ap(coco_evaluator)
    # ========================================

    # 打印自定义的场景评估结果
    calculate_and_print_custom_metrics(tp, fp, tn, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--weights', '-w', type=str, required=True, help="Path to the trained model weights (.pth).")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="Directory to save the visualized images. If None, images will not be saved.")
    args = parser.parse_args()
    dist_init()
    main(args)
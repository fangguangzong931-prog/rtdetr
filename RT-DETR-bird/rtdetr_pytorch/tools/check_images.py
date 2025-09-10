# 文件路径: check_images.py

import os
import json
from PIL import Image
from tqdm import tqdm

# --- 请根据你的实际情况修改这里的路径 ---
# 你数据集的根目录 (即包含 train2017, annotations 等文件夹的那个目录)
DATASET_ROOT = '/mnt/d/RT-DETR/RT-DETR-main/rtdetr_pytorch/dataset'

# 你的训练集标注文件名
ANNOTATION_FILE = '/mnt/d/RT-DETR/RT-DETR-main/rtdetr_pytorch/dataset/annotations/instances_train2017.json'


# -----------------------------------------


def check_dataset_images(dataset_root, annotation_file):
    """
    遍历数据集中的所有图片，检查是否有损坏的文件。
    """
    annotation_path = os.path.join(dataset_root, annotation_file)
    if not os.path.exists(annotation_path):
        print(f"错误：找不到标注文件 {annotation_path}")
        return

    print(f"正在加载标注文件: {annotation_path}")
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    images_info = data['images']
    corrupted_files = []

    print(f"开始检查 {len(images_info)} 张图片...")

    # 使用tqdm来显示进度条
    for img_info in tqdm(images_info):
        file_name = img_info['file_name']

        # 从文件名推断图片所在的文件夹 (通常是 train2017 或 val2017)
        # 这里的逻辑可能需要根据你的 .json 文件内容进行微调
        image_subfolder = os.path.dirname(file_name)
        if not image_subfolder:  # 如果file_name不包含子目录，则默认在train2017
            image_subfolder = 'train2017'  # 如果检查验证集，请改为 'val2017'
            file_name_only = file_name
        else:
            file_name_only = os.path.basename(file_name)

        img_path = os.path.join(dataset_root, image_subfolder, file_name_only)

        if not os.path.exists(img_path):
            print(f"\n警告：图片文件不存在 {img_path}")
            corrupted_files.append(file_name)
            continue

        try:
            # 尝试打开并验证图片
            with Image.open(img_path) as img:
                img.verify()  # verify()会检查文件是否损坏
        except Exception as e:
            # 如果打开或验证失败，记录为损坏文件
            print(f"\n发现损坏的图片: {img_path}")
            print(f"  错误信息: {e}")
            corrupted_files.append(file_name)

    if not corrupted_files:
        print("\n检查完成！所有图片均可正常读取。")
    else:
        print(f"\n检查完成！共发现 {len(corrupted_files)} 个损坏或缺失的图片文件：")
        for f in corrupted_files:
            print(f"- {f}")
        print("\n请从数据集中删除这些文件，并重新生成你的.json标注文件，然后再开始训练。")


if __name__ == '__main__':
    # 你可以修改这里的参数来检查不同的数据集（比如验证集）
    check_dataset_images(DATASET_ROOT, ANNOTATION_FILE)
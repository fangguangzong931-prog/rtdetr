# 文件路径: rtdetr_pytorch/tools/main.py
# (已修正 AttributeError)

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

# 从项目中导入必要的模块
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS

# 导入TensorBoard
from torch.utils.tensorboard import SummaryWriter


def main(args):
    """
    主函数，用于设置和启动训练流程
    """
    # 1. 初始化分布式环境和随机种子
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    # 2. 加载配置文件
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    # 3. 创建官方的Solver（训练器）实例
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # 4. 在外部给solver实例注入一个writer属性
    if dist.is_main_process():
        # === 修正区域 START ===
        # 错误原因：此时solver对象还没有.output_dir属性
        # 正确做法：直接从包含了所有配置的cfg对象中获取output_dir
        log_dir = os.path.join(cfg.output_dir, 'tensorboard_logs')
        # === 修正区域 END ===

        # 将 writer 附加到 solver 对象上
        solver.writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard日志已启动，请在训练开始后运行以下命令查看:")
        print(f"tensorboard --logdir={log_dir}\n")
    else:
        # 在其他进程上设置一个None，避免在写入时出错
        solver.writer = None

    # 5. 根据参数选择执行验证还是训练
    if args.test_only:
        solver.val()
    else:
        # 开始完整的训练流程
        solver.fit()

    # 6. 训练或验证结束后，如果创建了writer，则关闭它
    if dist.is_main_process() and hasattr(solver, 'writer') and solver.writer:
        solver.writer.close()
        print("TensorBoard writer has been closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 在这里设置默认参数，这样运行命令时就非常简洁
    parser.add_argument('--config', '-c', type=str,
                        default='/mnt/d/RT-DETR/RT-DETR-main/rtdetr_pytorch/configs/rtdetr/rtdetr_r50_bird.yml',
                        help='Path to the configuration file.')

    parser.add_argument('--resume', '-r', type=str,
                        default='',
                        help='Path to the checkpoint file for resuming training.')

    parser.add_argument('--tuning', '-t', type=str,
                        default='/mnt/d/RT-DETR/RT-DETR-main/weights/rtdetr_r50vd_6x_coco_from_paddle.pth',
                        help='Path to the pre-trained weights for fine-tuning.')

    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Only run evaluation on the validation set.')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (AMP).')

    parser.add_argument('--seed', type=int, default=None,
                        help='Set a random seed for reproducibility.')

    args = parser.parse_args()

    main(args)
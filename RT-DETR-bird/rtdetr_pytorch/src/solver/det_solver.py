# 文件路径: rtdetr_pytorch/src/solver/det_solver.py

'''
by lyuwenyu
'''
import time
import json
import datetime

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        # === 新增/修改区域 START ===
        # 初始化最佳模型的性能指标
        best_ap = 0.0
        # === 新增/修改区域 END ===

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # === 新增/修改区域 START ===
            # 将solver实例(self)传递给train_one_epoch，以便记录每一步的loss
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler, solver=self)
            # === 新增/修改区域 END ===

            self.lr_scheduler.step()

            # === 新增/修改区域 START ===
            # --- 保存逻辑 ---
            if self.output_dir:
                # 1. 保存最新的checkpoint (每轮都保存)
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'checkpoint.pth')

                # 2. 按用户要求，每10轮保存一次快照
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = self.output_dir / f'epoch_{epoch + 1}_checkpoint.pth'
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)
                    if dist.is_main_process():
                        print(f"定期快照已保存: {checkpoint_path}")
            # --- 保存逻辑结束 ---

            # --- 评估逻辑 ---
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # --- 最佳模型判断与保存 ---
            current_ap = test_stats['coco_eval_bbox'][0] # 获取 mAP[0.5:0.95]
            if current_ap > best_ap:
                best_ap = current_ap
                if self.output_dir:
                    dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best_checkpoint.pth')
                    if dist.is_main_process():
                        print(f"发现新的最佳模型！mAP: {best_ap:.4f}。已保存至 best_checkpoint.pth")

            # --- TensorBoard 日志记录 ---
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if dist.is_main_process() and hasattr(self, 'writer') and self.writer:
                # 记录训练loss (这是整个epoch的平均loss)
                self.writer.add_scalar('Loss/train_epoch_avg', train_stats['loss'], epoch)
                # 记录学习率
                self.writer.add_scalar('Misc/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
                # 记录评估指标
                self.writer.add_scalar('Metrics/mAP_0.5-0.95', test_stats['coco_eval_bbox'][0], epoch)
                self.writer.add_scalar('Metrics/mAP_0.5', test_stats['coco_eval_bbox'][1], epoch)
                self.writer.add_scalar('Metrics/AR_max100', test_stats['coco_eval_bbox'][8], epoch)
            # === 新增/修改区域 END ===

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
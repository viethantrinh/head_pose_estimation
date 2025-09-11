import inspect
import os
import shutil
import time
import torch
import yaml
import torch.nn.functional as F
import numpy as np

# Import both models
from model.ca_model_2 import Model as OriginalModel
from model.swin_1 import SwinHeadPoseEstimator

from utils.util import GradualWarmupScheduler, log, init_seed, worker_init_fn
from torch import optim
from feeder import dataset_p1 as dataset
from torch.utils.data import DataLoader
from utils.constant import *
from torch.backends import cudnn
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from utils.util import log


class SwinProcessor:
    """Enhanced processor that can use either original or Swin Transformer model"""

    def __init__(self, arg):
        self.arg = arg
        
        # Add model type selection
        self.use_swin = getattr(arg, 'use_swin', True)
        self.print_log(f"Using {'Swin Transformer' if self.use_swin else 'Original'} model")

        # 1. Fix the seed
        init_seed(0)

        # 2. Load the model
        self.load_model()

        # 3. Load the optimizer
        self.load_optimizer()

        # 4. Load the data
        self.load_data()

        # 5. Training parameters
        self.lr = self.arg.base_lr
        self.best_mae_test_2 = float("inf")
        self.best_mae_test1 = float("inf")
        self.best_epoch_test_2 = -1
        self.best_epoch_test1 = -1
        
        # Loss weights
        self.class_weight = 1.0
        self.reg_weight = 1.2
        self.beta1 = 1.0
        self.beta2 = 0.2
        self.alpha1 = 1.0
        self.alpha2 = 0.7
        self.temperature = 1.7

    def load_model(self):
        output_device = self.arg.device[0] if isinstance(
            self.arg.device, list) else self.arg.device
        self.output_device = output_device

        # Load Swin Transformer model
        self.model = SwinHeadPoseEstimator(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=66,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
        ).cuda(output_device)
        
        # Calculate and print model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f"Swin Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        self.print_log(f"Swin Model - Size: {total_params * 4 / (1024 * 1024):.2f} MB")
      

    def load_optimizer(self):
        """Initialize optimizer and scheduler from config."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.arg.base_lr,
            weight_decay=getattr(self.arg, 'weight_decay', 1e-4),
            betas=getattr(self.arg, 'betas', (0.9, 0.999)),
            eps=getattr(self.arg, 'eps', 1e-8)
        )

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=getattr(self.arg, 'step', [10, 20]),
            gamma=0.5
        )

        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            total_epoch=getattr(self.arg, 'warm_up_epoch', 5),
            after_scheduler=lr_scheduler_pre
        )

    def load_data(self):
        self.data_loader = {}

        dataset_map = {
            'Pose_300W_LP': dataset.Pose_300W_LP,
            'AFLW2000': dataset.AFLW2000,
            'BIWI': dataset.BIWI
        }

        # Initialize datasets
        train_dataset_class = dataset_map.get(self.arg.train_dataset)
        train_dataset = train_dataset_class(
            self.arg.train_data_path, self.arg.train_file_name)

        test_dataset1_class = dataset_map.get(self.arg.test_dataset1)
        test_dataset1 = test_dataset1_class(
            self.arg.test_data_path1, self.arg.test_file_name1)

        test_dataset2_class = dataset_map.get(self.arg.test_dataset2)
        test_dataset2 = test_dataset2_class(
            self.arg.test_data_path2, self.arg.test_file_name2)

        # Create data loaders
        self.data_loader[TRAIN_DATA] = DataLoader(
            dataset=train_dataset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed
        )

        self.data_loader[TEST_DATA_1] = DataLoader(
            dataset=test_dataset1,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )

        self.data_loader[TEST_DATA_2] = DataLoader(
            dataset=test_dataset2,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )

    def prepare_input(self, img):
        """Prepare input based on model type"""
       
        # Swin model expects full images, not patches
        return img


    def train(self, epoch, is_save_model=False):
        self.model.train()
        
        cudnn.enabled = True
        cudnn.deterministic = False
        cudnn.benchmark = True

        loader = self.data_loader[TRAIN_DATA]
        loss_value = []
        yaw_error, pitch_error, roll_error = 0.0, 0.0, 0.0
        total = 0
        idx_tensor = torch.arange(66, dtype=torch.float32).cuda(self.output_device)
        
        process = tqdm(loader, desc=f'Epoch {epoch + 1}/{self.arg.num_epoch}')
        
        for batch_idx, (img, euler_label, labels, index) in enumerate(process):
            batch_size = img.size(dim=0)
            total += batch_size

            img = Variable(img.float().cuda(self.output_device), requires_grad=True)
            euler_label = Variable(euler_label.float().cuda(self.output_device), requires_grad=False)
            labels = Variable(labels.long().cuda(self.output_device), requires_grad=False)

            # Prepare input based on model type
            model_input = self.prepare_input(img)

            # Forward pass
            yaw_class, pitch_class, roll_class = self.model(model_input)

            # Ground truth processing
            yaw_cont = torch.clamp(euler_label[:, 0], min=-99.0, max=99.0)
            pitch_cont = torch.clamp(euler_label[:, 1], min=-99.0, max=99.0)
            roll_cont = torch.clamp(euler_label[:, 2], min=-99.0, max=99.0)

            yaw_bin = labels[:, 0]
            pitch_bin = labels[:, 1]
            roll_bin = labels[:, 2]

            # Validate bin labels
            if (yaw_bin < 0).any() or (yaw_bin >= 66).any() or \
               (pitch_bin < 0).any() or (pitch_bin >= 66).any() or \
               (roll_bin < 0).any() or (roll_bin >= 66).any():
                self.print_log(f"Invalid bin labels at batch {batch_idx}")
                raise ValueError("Bin labels must be in range [0, 65]")

            # Calculate losses
            loss_yaw_class = self.beta1 * F.cross_entropy(yaw_class, yaw_bin)
            loss_pitch_class = self.beta2 * F.cross_entropy(pitch_class, pitch_bin)
            loss_roll_class = self.beta2 * F.cross_entropy(roll_class, roll_bin)
            class_loss = self.class_weight * (loss_yaw_class + loss_pitch_class + loss_roll_class)

            # Softmax and angle prediction
            yaw_pred_softmax = F.softmax(yaw_class / self.temperature, dim=1)
            pitch_pred_softmax = F.softmax(pitch_class / self.temperature, dim=1)
            roll_pred_softmax = F.softmax(roll_class / self.temperature, dim=1)

            yaw_pred = 3 * torch.sum(yaw_pred_softmax * idx_tensor, dim=1) - 99
            pitch_pred = 3 * torch.sum(pitch_pred_softmax * idx_tensor, dim=1) - 99
            roll_pred = 3 * torch.sum(roll_pred_softmax * idx_tensor, dim=1) - 99

            # Regression loss
            loss_yaw_wrapped = self.alpha1 * self.wrapped_loss(yaw_pred, yaw_cont)
            loss_pitch_wrapped = self.alpha2 * self.wrapped_loss(pitch_pred, pitch_cont)
            loss_roll_wrapped = self.alpha2 * self.wrapped_loss(roll_pred, roll_cont)
            reg_loss = self.reg_weight * (loss_yaw_wrapped + loss_pitch_wrapped + loss_roll_wrapped)

            total_loss = class_loss + reg_loss

            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track errors
            yaw_error += torch.sum(torch.abs(yaw_pred - yaw_cont)).item()
            pitch_error += torch.sum(torch.abs(pitch_pred - pitch_cont)).item()
            roll_error += torch.sum(torch.abs(roll_pred - roll_cont)).item()

            loss_value.append(total_loss.item())
            process.set_postfix({
                'loss': f'{np.mean(loss_value):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Print training statistics
        mean_loss = np.mean(loss_value)
        mean_yaw_error = yaw_error / total
        mean_pitch_error = pitch_error / total
        mean_roll_error = roll_error / total

        self.print_log(f'\tMean training loss: {mean_loss:.4f}')
        self.print_log(f'\tLearning rate: {self.lr:.6f}')
        self.print_log(f'\tYaw MAE: {mean_yaw_error:.4f}')
        self.print_log(f'\tPitch MAE: {mean_pitch_error:.4f}')
        self.print_log(f'\tRoll MAE: {mean_roll_error:.4f}')

        self.lr_scheduler.step(epoch)

        if is_save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()]
                                  for k, v in state_dict.items()])
            model_name = "swin" if self.use_swin else "original"
            torch.save(weights, f'{self.arg.model_saved_name}-{model_name}-{epoch}.pt')

    def eval(self, epoch, loader_name=['test1', 'test2']):
        self.model.eval()
        self.print_log("=" * 86)
        
        for ln in loader_name:
            loss_value = []
            yaw_error, pitch_error, roll_error = 0.0, 0.0, 0.0
            total = 0

            idx_tensor = torch.arange(66, dtype=torch.float32).cuda(self.output_device)

            process = tqdm(self.data_loader[ln], desc=f'Eval {ln} - Epoch {epoch + 1}')
            
            for batch_idx, (img, euler_label, labels, index) in enumerate(process):
                batch_size = img.size(0)
                total += batch_size
                
                with torch.inference_mode():
                    img = Variable(img.float().cuda(self.output_device))
                    euler_label = Variable(euler_label.float().cuda(self.output_device))
                    labels = Variable(labels.long().cuda(self.output_device))

                    model_input = self.prepare_input(img)
                    yaw_class, pitch_class, roll_class = self.model(model_input)

                    # Ground truth processing
                    yaw_cont = torch.clamp(euler_label[:, 0], min=-99.0, max=99.0)
                    pitch_cont = torch.clamp(euler_label[:, 1], min=-99.0, max=99.0)
                    roll_cont = torch.clamp(euler_label[:, 2], min=-99.0, max=99.0)

                    yaw_bin = labels[:, 0]
                    pitch_bin = labels[:, 1]
                    roll_bin = labels[:, 2]

                    # Skip invalid batches
                    if (yaw_bin < 0).any() or (yaw_bin >= 66).any() or \
                       (pitch_bin < 0).any() or (pitch_bin >= 66).any() or \
                       (roll_bin < 0).any() or (roll_bin >= 66).any():
                        continue

                    # Calculate losses (same as training)
                    loss_yaw_class = self.beta1 * F.cross_entropy(yaw_class, yaw_bin)
                    loss_pitch_class = self.beta2 * F.cross_entropy(pitch_class, pitch_bin)
                    loss_roll_class = self.beta2 * F.cross_entropy(roll_class, roll_bin)
                    class_loss = self.class_weight * (loss_yaw_class + loss_pitch_class + loss_roll_class)

                    yaw_pred_softmax = F.softmax(yaw_class / self.temperature, dim=1)
                    pitch_pred_softmax = F.softmax(pitch_class / self.temperature, dim=1)
                    roll_pred_softmax = F.softmax(roll_class / self.temperature, dim=1)

                    yaw_pred = 3 * torch.sum(yaw_pred_softmax * idx_tensor, dim=1) - 99
                    pitch_pred = 3 * torch.sum(pitch_pred_softmax * idx_tensor, dim=1) - 99
                    roll_pred = 3 * torch.sum(roll_pred_softmax * idx_tensor, dim=1) - 99

                    loss_yaw_wrapped = self.alpha1 * self.wrapped_loss(yaw_pred, yaw_cont)
                    loss_pitch_wrapped = self.alpha2 * self.wrapped_loss(pitch_pred, pitch_cont)
                    loss_roll_wrapped = self.alpha2 * self.wrapped_loss(roll_pred, roll_cont)
                    reg_loss = self.reg_weight * (loss_yaw_wrapped + loss_pitch_wrapped + loss_roll_wrapped)

                    total_loss = class_loss + reg_loss

                    yaw_error += torch.sum(torch.abs(yaw_pred - yaw_cont)).item()
                    pitch_error += torch.sum(torch.abs(pitch_pred - pitch_cont)).item()
                    roll_error += torch.sum(torch.abs(roll_pred - roll_cont)).item()

                    loss_value.append(total_loss.item())

            # Calculate and print statistics
            mean_loss = np.mean(loss_value)
            mean_yaw_error = yaw_error / total
            mean_pitch_error = pitch_error / total
            mean_roll_error = roll_error / total
            mean_mae = (mean_yaw_error + mean_pitch_error + mean_roll_error) / 3

            self.print_log(f'\tMAE {ln} loss: {mean_mae:.4f}')
            self.print_log(f'\tMAE yaw {ln}: {mean_yaw_error:.4f}')
            self.print_log(f'\tMAE pitch {ln}: {mean_pitch_error:.4f}')
            self.print_log(f'\tMAE roll {ln}: {mean_roll_error:.4f}')

            # Save best models
            if ln == 'test1' and mean_mae < self.best_mae_test1:
                self.best_mae_test1 = mean_mae
                self.best_epoch_test1 = epoch
                self.save_best_model('test1', epoch)

            if ln == 'test2' and mean_mae < self.best_mae_test_2:
                self.best_mae_test_2 = mean_mae
                self.best_epoch_test_2 = epoch
                self.save_best_model('test2', epoch)

        # Print best results summary
        self.print_log("=" * 86)
        self.print_log("BEST RESULTS SUMMARY:")
        self.print_log(f'\tBest Test1 MAE: {self.best_mae_test1:.4f} at epoch {self.best_epoch_test1 + 1}')
        self.print_log(f'\tBest Test2 MAE: {self.best_mae_test_2:.4f} at epoch {self.best_epoch_test_2 + 1}')
        self.print_log("=" * 86)

    def save_best_model(self, test_name, epoch):
        """Save best model with appropriate naming"""
        model_type = "swin" if self.use_swin else "original"
        self.print_log(f'\tSaving best {model_type} model for {test_name} at epoch {epoch} with MAE: {getattr(self, f"best_mae_{test_name.replace("test", "test_")}"):.4f}')
        
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        
        filename = f'best_model_{test_name}_{model_type}.pt'
        torch.save(weights, os.path.join(self.arg.work_dir, filename))

    def wrapped_loss(self, pred, target):
        """Compute wrapped loss for cyclic angles."""
        diff = pred - target
        abs_diff = torch.abs(diff)
        min_diff = torch.min(abs_diff, 360 - abs_diff)
        return torch.mean(min_diff ** 2)

    def print_log(self, str, print_time=True):
        """Print log and write to file."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = f"[ {localtime} ] {str}"

        print(str)

        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(str, file=f)

    def split_image(self, x):
        """Split image into 4 patches (for original model compatibility)."""
        bs, c, h, w = x.shape
        patch_size = h // 2
        patches = []
        for i in range(2):
            for j in range(2):
                patch = x[:, :, i * patch_size:(i + 1) * patch_size,
                          j * patch_size:(j + 1) * patch_size]
                patches.append(patch)
        return patches

    def start(self):
        self.print_log(f"Start training from epoch {self.arg.start_epoch + 1} to {self.arg.num_epoch}")
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            is_save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            self.train(epoch, is_save_model=is_save_model)
            self.eval(epoch)
            torch.cuda.empty_cache()
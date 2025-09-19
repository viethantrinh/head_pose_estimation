import inspect
import os
import shutil
import time
import torch
import yaml
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import random

from model.bi_directional_cross_attention.ca_model_all_stable_light import Model
from utils.util import GradualWarmupScheduler, log, init_seed, worker_init_fn
from torch import optim
from feeder import dataset_p2 as dataset
from torch.utils.data import DataLoader
from utils.constant import *
from torch.backends import cudnn
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from utils.util import log


class ProcessorTest:
    """
    Processor for Head Pose Estimation (Testing) - Protocol 2
    """

    def __init__(self, arg):
        self.arg = arg

        # 1. Fix the seed to 1

        # 2. Save the current information of config file into a file
        # self.save_train_config_file()

        # 3. Load the model
        self.load_model()

        # 4. Load the data from dataset
        self.load_data()

        # 5. Define the best MAE for test1 (BIWI_test)
        self.best_mae_test1 = float("inf")

        # 6. Best epoch tracking
        self.best_epoch_test1 = -1

        # 7. Define the class weight
        self.class_weight = 1.0

        # 8. Define the regression weight
        self.reg_weight = 1.2

        # 9. Define the weight for yaw
        self.beta1 = 1.0

        # 10. Define the weight for pitch and roll
        self.beta2 = 0.2

        # 11. Define the Weight for yaw regression
        self.alpha1 = 1.0

        # 12. Define the weight for pitch and roll regression
        self.alpha2 = 0.7

        # 13. Define the stability for soft-argmax
        self.temperature = 1.7  # For soft-argmax stability

    def load_model(self):
        """Initialize the model and load weights."""
        output_device = self.arg.device[0] if isinstance(
            self.arg.device, list) else self.arg.device
        self.output_device = output_device  # select gpu

        self.model = Model().cuda(output_device)  # initialize the model to GPU

        # Load weights if specified
        if hasattr(self.arg, 'weights_file_extract') and self.arg.weights_file_extract:
            weights_file = self.arg.weights_file_extract
            if not os.path.exists(weights_file):
                self.print_log(f"Weights file not found: {weights_file}")
                sys.exit(1)
            if os.path.getsize(weights_file) == 0:
                self.print_log(f"Weights file is empty: {weights_file}")
                sys.exit(1)

            try:
                weights = torch.load(
                    weights_file, map_location=lambda storage, loc: storage.cuda(self.output_device))
                if not isinstance(weights, (dict, OrderedDict)):
                    self.print_log(
                        f"Invalid weights format in {weights_file}. Expected a dictionary.")
                    sys.exit(1)
                weights = OrderedDict(
                    [[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
                self.model.load_state_dict(weights)
                self.print_log(
                    f"Successfully loaded weights from {weights_file}")
            except Exception as e:
                self.print_log(
                    f"Error loading weights from {weights_file}: {str(e)}")
                self.print_log("Suggestions:")
                self.print_log(
                    "- Verify the file integrity (e.g., re-save or download again).")
                self.print_log(
                    "- Check PyTorch version compatibility (try loading with the same version used to save).")
                self.print_log(
                    "- Ensure the file is not corrupted (file size > 0 and not locked).")
                sys.exit(1)

    def load_data(self):
        """Load testing data for Protocol 2."""
        self.data_loader = {}

        # Protocol 2 only uses BIWI dataset with train/test split
        # Initialize the dataset class for testing
        test_dataset1 = dataset.BIWI_test(
            self.arg.test_data_path1, self.arg.test_file_name1)

        # Set data loader for test data
        self.data_loader[TEST_DATA_1] = DataLoader(
            dataset=test_dataset1,
            batch_size=self.arg.test_batch_size,
            shuffle=False,  # test => don't need to shuffle
            num_workers=self.arg.num_worker,
            drop_last=False,  # test all the data, no need drop last
            worker_init_fn=init_seed
        )

        # Set some config for faster computation and reproducibility
        cudnn.enabled = True
        cudnn.deterministic = False
        cudnn.benchmark = True

    def eval(self, epoch=0, loader_name=[TEST_DATA_1]):
        """Evaluate the model on test dataset (Protocol 2)."""
        self.model.eval()
        self.print_log(
            "======================================================================================")
        for ln in loader_name:
            if ln not in self.data_loader:
                self.print_log(
                    f'Warning: No data loader found for {ln}. Skipping evaluation for {ln}.')
                continue

            loss_value = []
            yaw_error, pitch_error, roll_error = 0.0, 0.0, 0.0
            total = 0

            idx_tensor = torch.arange(
                66, dtype=torch.float32).cuda(self.output_device)

            process = tqdm(
                self.data_loader[ln], desc=f'Eval {ln} - Testing')
            for batch_idx, (img, euler_label, labels, index) in enumerate(process):
                try:
                    batch_size = img.size(0)
                    total += batch_size
                    with torch.inference_mode():
                        img = Variable(img.float().cuda(self.output_device))
                        euler_label = Variable(
                            euler_label.float().cuda(self.output_device))
                        labels = Variable(
                            labels.long().cuda(self.output_device))

                        x_parts = self.split_image(img)
                        yaw_class, pitch_class, roll_class = self.model(
                            x_parts)

                        # Clamp angles to [-99, 99] (as per dataset)
                        yaw_cont = torch.clamp(
                            euler_label[:, 0], min=-99.0, max=99.0)
                        pitch_cont = torch.clamp(
                            euler_label[:, 1], min=-99.0, max=99.0)
                        roll_cont = torch.clamp(
                            euler_label[:, 2], min=-99.0, max=99.0)

                        # Use labels (bin labels) from dataset
                        yaw_bin = labels[:, 0]
                        pitch_bin = labels[:, 1]
                        roll_bin = labels[:, 2]

                        # Check bin labels
                        if (yaw_bin < 0).any() or (yaw_bin >= 66).any() or (pitch_bin < 0).any() or (pitch_bin >= 66).any() or (
                                roll_bin < 0).any() or (roll_bin >= 66).any():
                            self.print_log(
                                f"Warning: Invalid bin labels at batch {batch_idx} in {ln}, skipping batch.")
                            continue

                        # Cross-Entropy Loss
                        loss_yaw_class = self.beta1 * \
                            F.cross_entropy(yaw_class, yaw_bin)
                        loss_pitch_class = self.beta2 * \
                            F.cross_entropy(pitch_class, pitch_bin)
                        loss_roll_class = self.beta2 * \
                            F.cross_entropy(roll_class, roll_bin)
                        class_loss = self.class_weight * \
                            (loss_yaw_class + loss_pitch_class + loss_roll_class)

                        # Soft-argmax with temperature scaling
                        yaw_pred_softmax = F.softmax(
                            yaw_class / self.temperature, dim=1)
                        pitch_pred_softmax = F.softmax(
                            pitch_class / self.temperature, dim=1)
                        roll_pred_softmax = F.softmax(
                            roll_class / self.temperature, dim=1)

                        # Compute angle predictions using Arg_softmax formula
                        yaw_pred = 3 * \
                            torch.sum(yaw_pred_softmax *
                                      idx_tensor, dim=1) - 99
                        pitch_pred = 3 * \
                            torch.sum(pitch_pred_softmax *
                                      idx_tensor, dim=1) - 99
                        roll_pred = 3 * \
                            torch.sum(roll_pred_softmax *
                                      idx_tensor, dim=1) - 99

                        # Wrapped Loss
                        loss_yaw_wrapped = self.alpha1 * \
                            self.wrapped_loss(yaw_pred, yaw_cont)
                        loss_pitch_wrapped = self.alpha2 * \
                            self.wrapped_loss(pitch_pred, pitch_cont)
                        loss_roll_wrapped = self.alpha2 * \
                            self.wrapped_loss(roll_pred, roll_cont)
                        reg_loss = self.reg_weight * \
                            (loss_yaw_wrapped + loss_pitch_wrapped + loss_roll_wrapped)

                        # Total loss
                        total_loss = class_loss + reg_loss

                        yaw_error += torch.sum(torch.abs(yaw_pred -
                                               yaw_cont)).item()
                        pitch_error += torch.sum(torch.abs(pitch_pred -
                                                 pitch_cont)).item()
                        roll_error += torch.sum(torch.abs(roll_pred -
                                                roll_cont)).item()

                        loss_value.append(total_loss.item())

                except Exception as e:
                    self.print_log(
                        f"Error processing batch {batch_idx} in {ln}: {str(e)}")
                    continue

            if total == 0:
                self.print_log(
                    f"No valid batches processed for {ln}, skipping evaluation.")
                continue

            mean_loss = np.mean(loss_value) if loss_value else 0.0
            mean_yaw_error = yaw_error / total
            mean_pitch_error = pitch_error / total
            mean_roll_error = roll_error / total
            mean_mae = (mean_yaw_error + mean_pitch_error +
                        mean_roll_error) / 3

            self.print_log(
                f'\tMAE {ln} loss (avg over {len(self.data_loader[ln])} batches): {mean_mae:.4f}')
            self.print_log(f'\tMAE yaw {ln}: {mean_yaw_error:.4f}')
            self.print_log(f'\tMAE pitch {ln}: {mean_pitch_error:.4f}')
            self.print_log(f'\tMAE roll {ln}: {mean_roll_error:.4f}')

            # Update best results for tracking
            if ln == TEST_DATA_1 and mean_mae < self.best_mae_test1:
                self.best_mae_test1 = mean_mae
                self.best_epoch_test1 = epoch
                self.print_log(
                    f'\tNew best result for {ln} with MAE: {self.best_mae_test1:.4f}')

        # Print results summary
        self.print_log(
            "======================================================================================")
        self.print_log("TESTING RESULTS SUMMARY (Protocol 2):")
        self.print_log(
            f'\tTest ({self.arg.test_dataset1}) MAE: {self.best_mae_test1:.4f}')
        self.print_log(
            "======================================================================================")

    def wrapped_loss(self, pred, target):
        """Compute wrapped loss for cyclic angles using 360-degree cycle, as described in the paper."""
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

        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            try:
                with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                    print(str, file=f)
            except Exception as e:
                print(f"Error writing to log file: {str(e)}")

    def split_image(self, x):
        """Split image into 4 patches."""
        bs, c, h, w = x.shape  # bs is batch_size, c is channel, h is height, w is width
        patch_size = h // 2
        patches = []
        for i in range(2):
            for j in range(2):
                patch = x[:, :, i * patch_size:(i + 1) * patch_size,
                          j * patch_size:(j + 1) * patch_size]
                patches.append(patch)
        return patches

    def start(self):
        """Start the evaluation process."""
        if hasattr(self.arg, 'weights_file_extract') and self.arg.weights_file_extract:
            self.print_log(
                f'Start evaluation with weights from {self.arg.weights_file_extract}')
        else:
            self.print_log(
                'Start evaluation with randomly initialized weights')

        self.eval(epoch=0, loader_name=[TEST_DATA_1])

        self.print_log(
            f'Best MAE achieved: {self.best_mae_test1:.4f} degrees')
        self.print_log('Evaluation completed.\n')

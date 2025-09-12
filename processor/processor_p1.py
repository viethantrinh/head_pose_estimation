import inspect
import os
import shutil
import time
import torch
import yaml
import torch.nn.functional as F
import numpy as np

from model.PE_change import *
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


class Processor:

    def __init__(self, arg):

        self.arg = arg

        # 1. Fix the seed to 1
        

        # 2. Save the current information of config file into a file
        # self.save_train_config_file()

        # 4. Load the model
        self.load_model()

        # 5. Load the optimizer
        self.load_optimizer()

        # 6. Load the data from dataset
        self.load_data()

        # 7. Define the learning rate
        self.lr = self.arg.base_lr

        # 8. Define the best MAE for test2
        self.best_mae_test_2 = float("inf")

        # 8.1. Define the best MAE for test1
        self.best_mae_test1 = float("inf")

        # 9. Define the best epoch for test 2
        self.best_epoch_test_2 = -1

        # 9.1. Define the best epoch for test1
        self.best_epoch_test1 = -1

        # 10. Define the class weight
        self.class_weight = 1.0

        # 11. Define the regression weight
        self.reg_weight = 1.2

        # 12. Define the weight for yaw
        self.beta1 = 1.0

        # 13. Define the weight for pitch and roll
        self.beta2 = 0.2

        # 14. Define the Weight for yaw regression
        self.alpha1 = 1.0

        # 15. Define the weight for pitch and roll regression
        self.alpha2 = 0.7

        # 16. Define the stability for soft-argmax
        self.temperature = 1.7  # For soft-argmax stability

    def save_train_config_file(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.arg), f)

    def load_model(self):
        output_device = self.arg.device[0] if isinstance(
            self.arg.device, list) else self.arg.device
        self.output_device = output_device  # select gpu

        # save the model file
        # shutil.copy2(inspect.getfile(Model), self.arg.work_dir)

        self.model = Model().cuda(output_device)  # initialize the model to GPU

    def load_optimizer(self):
        """Initialize optimizer and scheduler from config."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.arg.base_lr,
            weight_decay=self.arg.weight_decay if hasattr(
                self.arg, 'weight_decay') else 1e-4,
            betas=self.arg.betas if hasattr(
                self.arg, 'betas') else (0.9, 0.999),
            eps=self.arg.eps if hasattr(self.arg, 'eps') else 1e-8
        )

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.arg.step if hasattr(
                self.arg, 'step') else [10, 20],
            gamma=0.5
        )

        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            total_epoch=self.arg.warm_up_epoch if hasattr(
                self.arg, 'warm_up_epoch') else 5,
            after_scheduler=lr_scheduler_pre
        )

        self.print_log(
            f'Using warm-up scheduler, total warm-up epochs: {self.arg.warm_up_epoch if hasattr(self.arg, "warm_up_epoch") else 5}')

    def load_data(self):
        self.data_loader = {}

        dataset_map = {
            'Pose_300W_LP': dataset.Pose_300W_LP,
            'AFLW2000': dataset.AFLW2000,
            'BIWI': dataset.BIWI
        }

        # initialize the dataset class for training and testing
        train_dataset_class = dataset_map.get(self.arg.train_dataset)
        train_dataset = train_dataset_class(
            self.arg.train_data_path, self.arg.train_file_name)

        test_dataset1_class = dataset_map.get(self.arg.test_dataset1)
        test_dataset1 = test_dataset1_class(
            self.arg.test_data_path1, self.arg.test_file_name1)

        test_dataset2_class = dataset_map.get(self.arg.test_dataset2)
        test_dataset2 = test_dataset2_class(
            self.arg.test_data_path2, self.arg.test_file_name2)

        # set each key for the train and the test data
        self.data_loader[TRAIN_DATA] = DataLoader(
            dataset=train_dataset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            # the amount of parallel thread used to load the data
            num_workers=self.arg.num_worker,
            drop_last=True,  # if the batch size is not enough cover then ignore
            worker_init_fn=init_seed
        )

        self.data_loader[TEST_DATA_1] = DataLoader(
            dataset=test_dataset1,
            batch_size=self.arg.test_batch_size,
            shuffle=False,  # test => don't need to shuffle
            num_workers=self.arg.num_worker,
            drop_last=False,  # test all the data, no need drop last
            worker_init_fn=init_seed
        )

        self.data_loader[TEST_DATA_2] = DataLoader(
            dataset=test_dataset2,
            batch_size=self.arg.test_batch_size,
            shuffle=False,  # test => don't need to shuffle
            num_workers=self.arg.num_worker,
            drop_last=False,  # test all the data, no need drop last
            worker_init_fn=init_seed
        )

    def train(self, epoch, is_save_model=False):
        self.model.train()
        
        # Set some config for faster computation and reproducibility
        cudnn.enabled = True
        cudnn.deterministic = False
        cudnn.benchmark = True

        loader = self.data_loader[TRAIN_DATA]
        loss_value = []
        yaw_error, pitch_error, roll_error = 0.0, 0.0, 0.0
        total = 0
        idx_tensor = torch.arange(
            66, dtype=torch.float32).cuda(self.output_device)
        
        process = tqdm(loader, desc=f'Epoch {epoch + 1}/{self.arg.num_epoch}')
        
        for batch_idx, (img, euler_label, labels, index) in enumerate(process):
            # [dim0, dim1, dim2, dim3] - [number_of_samples (batch_size), channel, height, width]
            batch_size = img.size(dim=0)
            total += batch_size

            img = Variable(img.float().cuda(
                self.output_device), requires_grad=True)
            euler_label = Variable(euler_label.float().cuda(
                self.output_device), requires_grad=False)
            labels = Variable(labels.long().cuda(
                self.output_device), requires_grad=False)

            # 1. split image into 4 parts
            # [ [...], [...], [...], [...] ] -> each tensor's size = (100, 3, 32, 32) - x_parts is list
            x_parts = self.split_image(img)

            # ================================================================================================
            # ======================== Forward ===============================================================
            # ================================================================================================
            # 2. Forward into model and get the fully connected values
            # all is having shape (100, 66) - 100 is batch size, 66 is the bin
            yaw_class, pitch_class, roll_class = self.model(x_parts)

            # ================================================================================================
            # ======================== Ground truth ==========================================================
            # ================================================================================================
            # 3. Clamp angles to [-99, 99] (as per dataset) (ground truth yaw, pich, roll, angle's labels)
            # This step take a continuous real angles number of yaw, pitch and roll in [-99 degree, 99 degree]
            yaw_cont = torch.clamp(euler_label[:, 0], min=-99.0, max=99.0)
            pitch_cont = torch.clamp(euler_label[:, 1], min=-99.0, max=99.0)
            roll_cont = torch.clamp(euler_label[:, 2], min=-99.0, max=99.0)

            # 4. Take labels bin ground truth (bin nay duoc chuyen sang tu angles goc cua yaw, pitch va roll)
            yaw_bin = labels[:, 0]
            pitch_bin = labels[:, 1]
            roll_bin = labels[:, 2]

            # Check if the bin is valid or not
            if (yaw_bin < 0).any() or (yaw_bin >= 66).any() or (pitch_bin < 0).any() or (pitch_bin >= 66).any() or (roll_bin < 0).any() or (roll_bin >= 66).any():
                self.print_log(
                    f"Invalid bin labels at batch {batch_idx}: yaw_bin={yaw_bin}, pitch_bin={pitch_bin}, roll_bin={roll_bin}")
                raise ValueError("Bin labels must be in range [0, 65]")

            # ================================================================================================
            # ======================== Calculate the angle the bin and the loss ====================================================
            # ================================================================================================
            # 5. =>>>>>>>>>>> Cross-Entropy Loss (for bin classification) - calculate the bin loss
            loss_yaw_class = self.beta1 * F.cross_entropy(yaw_class, yaw_bin)
            loss_pitch_class = self.beta2 * \
                F.cross_entropy(pitch_class, pitch_bin)
            loss_roll_class = self.beta2 * \
                F.cross_entropy(roll_class, roll_bin)
            class_loss = self.class_weight * \
                (loss_yaw_class + loss_pitch_class + loss_roll_class)

            # 6. Go through softmax function to output the probability of each head (pi)
            yaw_pred_softmax = F.softmax(yaw_class / self.temperature, dim=1)
            pitch_pred_softmax = F.softmax(
                pitch_class / self.temperature, dim=1)
            roll_pred_softmax = F.softmax(roll_class / self.temperature, dim=1)

            # 7. Calculate the angle prediction using arg_softmax formula - tinh goc du doan dua tren ham softmax
            yaw_pred = 3 * torch.sum(yaw_pred_softmax * idx_tensor, dim=1) - 99
            pitch_pred = 3 * \
                torch.sum(pitch_pred_softmax * idx_tensor, dim=1) - 99
            roll_pred = 3 * \
                torch.sum(roll_pred_softmax * idx_tensor, dim=1) - 99

            # 8. =>>>>>>>>>>>>>> Wrapped loss for angle regression - calculate the angle loss
            loss_yaw_wrapped = self.alpha1 * \
                self.wrapped_loss(yaw_pred, yaw_cont)
            loss_pitch_wrapped = self.alpha2 * \
                self.wrapped_loss(pitch_pred, pitch_cont)
            loss_roll_wrapped = self.alpha2 * \
                self.wrapped_loss(roll_pred, roll_cont)
            reg_loss = self.reg_weight * \
                (loss_yaw_wrapped + loss_pitch_wrapped + loss_roll_wrapped)

            total_loss = class_loss + reg_loss

            # ================================================================================================
            # ======================== Training and optimize ====================================================
            # ================================================================================================
            # 9. zero grad
            self.optimizer.zero_grad()

            # 10. loss backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            # 11. optimize step
            self.optimizer.step()

            yaw_error += torch.sum(torch.abs(yaw_pred - yaw_cont)).item()
            pitch_error += torch.sum(torch.abs(pitch_pred - pitch_cont)).item()
            roll_error += torch.sum(torch.abs(roll_pred - roll_cont)).item()

            loss_value.append(total_loss.item())
            process.set_postfix({'loss': f'{np.mean(loss_value):.4f}',
                                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})

        mean_loss = np.mean(loss_value)
        mean_yaw_error = yaw_error / total  # total batchs
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
            torch.save(weights, f'{self.arg.model_saved_name}-{epoch}.pt')

    def eval(self, epoch, loader_name=['test1', 'test2']):
        self.model.eval()
        self.print_log(
            "======================================================================================")
        for ln in loader_name:
            loss_value = []
            yaw_error, pitch_error, roll_error = 0.0, 0.0, 0.0
            total = 0

            idx_tensor = torch.arange(
                66, dtype=torch.float32).cuda(self.output_device)

            process = tqdm(
                self.data_loader[ln], desc=f'Eval {ln} - Epoch {epoch + 1}')
            for batch_idx, (img, euler_label, labels, index) in enumerate(process):
                batch_size = img.size(0)
                total += batch_size
                with torch.inference_mode():
                    img = Variable(img.float().cuda(self.output_device))
                    euler_label = Variable(
                        euler_label.float().cuda(self.output_device))
                    labels = Variable(labels.long().cuda(self.output_device))

                    x_parts = self.split_image(img)
                    yaw_class, pitch_class, roll_class = self.model(x_parts)

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
                            f"Skipping batch {batch_idx} in {ln} due to invalid bin labels: yaw={yaw_bin}, pitch={pitch_bin}, roll={roll_bin}")
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
                        torch.sum(yaw_pred_softmax * idx_tensor, dim=1) - 99
                    pitch_pred = 3 * \
                        torch.sum(pitch_pred_softmax * idx_tensor, dim=1) - 99
                    roll_pred = 3 * \
                        torch.sum(roll_pred_softmax * idx_tensor, dim=1) - 99

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

            mean_loss = np.mean(loss_value)
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

            # Save best model for test1
            if ln == 'test1' and mean_mae < self.best_mae_test1:
                self.best_mae_test1 = mean_mae
                self.best_epoch_test1 = epoch
                self.print_log(
                    f'\tSaving best model for test1 at epoch {self.best_epoch_test1} with MAE: {self.best_mae_test1:.4f}')
                state_dict = self.model.state_dict()
                weights = OrderedDict(
                    [[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(weights, os.path.join(
                    self.arg.work_dir, 'best_model_test1.pt'))

            # Save best model for test2
            if ln == 'test2' and mean_mae < self.best_mae_test_2:
                self.best_mae_test_2 = mean_mae
                self.best_epoch_test_2 = epoch
                self.print_log(
                    f'\tSaving best model for test2 at epoch {self.best_epoch_test_2} with MAE: {self.best_mae_test_2:.4f}')
                state_dict = self.model.state_dict()
                weights = OrderedDict(
                    [[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(weights, os.path.join(
                    self.arg.work_dir, 'best_model_test2.pt'))

        # Print best results summary after evaluating both test sets
        self.print_log(
            "======================================================================================")
        self.print_log("BEST RESULTS SUMMARY:")
        self.print_log(
            f'\tBest Test1 MAE: {self.best_mae_test1:.4f} at epoch {self.best_epoch_test1 + 1}')
        self.print_log(
            f'\tBest Test2 MAE: {self.best_mae_test_2:.4f} at epoch {self.best_epoch_test_2 + 1}')
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

        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(str, file=f)

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
        self.print_log(
            f"Start training from epoch {self.arg.start_epoch + 1} to {self.arg.num_epoch}")
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            is_save_model = ((epoch + 1) % self.arg.save_interval ==
                             0) or (epoch + 1 == self.arg.num_epoch)
            self.train(epoch, is_save_model=is_save_model)
            self.eval(epoch)
            torch.cuda.empty_cache()

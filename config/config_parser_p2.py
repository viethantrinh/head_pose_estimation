import argparse
from utils.util import str2bool


def get_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head Pose Estimation')

    # =============================================================================
    # General Configuration
    # =============================================================================
    parser.add_argument(
        '--config',
        default='./config/train_config_p2.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/p2',
        help='the work folder for storing results')

    # =============================================================================
    # Model Configuration
    # =============================================================================
    parser.add_argument(
        '--model-saved-name',
        default='',
        help='name for saving the model')
    parser.add_argument(
        '--weights-file-extract',
        type=str,
        default="",
        help='extract weights from file')

    # =============================================================================
    # Training Dataset Configuration
    # =============================================================================
    parser.add_argument(
        '--train-dataset',
        help='Training dataset type.',
        default='BIWI_train',
        type=str)
    parser.add_argument(
        '--train-data-path',
        help='Directory path for training data.',
        default='./data/resize_BIWI',
        type=str)
    parser.add_argument(
        '--train-file-name',
        help='Path to text file containing relative paths for training examples.',
        default='./file_list/protocol_2/BIWI_train.txt',
        type=str)

    # =============================================================================
    # Test Dataset Configuration
    # =============================================================================
    # Test Dataset 1
    parser.add_argument(
        '--test-dataset1',
        help='Test dataset 1 type.',
        default='BIWI_test',
        type=str)
    parser.add_argument(
        '--test-data-path1',
        help='Directory path for test data 1.',
        default='./data/resize_BIWI',
        type=str)
    parser.add_argument(
        '--test-file-name1',
        help='Path to text file containing relative paths for test examples 1.',
        default='./file_list/protocol_2/BIWI_test.txt',
        type=str)

    # =============================================================================
    # Training Hyperparameters
    # =============================================================================
    # Learning rate and optimization
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 50, 80],
        nargs='+',
        help='the epoch where optimizer reduces the learning rate')

    # Batch sizes and epochs
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=128,
        help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=100,
        help='stop training in which epoch')
    parser.add_argument(
        '--warm-up-epoch',
        type=int,
        default=0,
        help='number of warmup epochs')

    # =============================================================================
    # System Configuration
    # =============================================================================
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of data loading workers')

    # =============================================================================
    # Logging and Evaluation
    # =============================================================================
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')

    return parser

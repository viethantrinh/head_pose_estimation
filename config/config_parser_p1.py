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
        default='./config/train_config_p1.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/p1',
        help='the work folder for storing results')
    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     default=1,
    #     help='random seed for pytorch')
    # parser.add_argument(
    #     '--debug',
    #     type=str2bool,
    #     default=False,
    #     help='enable debug mode')

    # =============================================================================
    # Model Configuration
    # =============================================================================
    # parser.add_argument(
    #     '--model',
    #     default=None,
    #     help='the model will be used')
    # parser.add_argument(
    #     '--model-args',
    #     type=dict,
    #     default=dict(),
    #     help='the arguments of model')
    parser.add_argument(
        '--model-saved-name',
        default='',
        help='name for saving the model')

    # Model weights and checkpoints
    # parser.add_argument(
    #     '--weights',
    #     default=None,
    #     help='the weights for network initialization')
    # parser.add_argument(
    #     '--weights-file',
    #     default='',
    #     help='model file path')
    parser.add_argument(
        '--weights-file-extract',
        type=str,
        default="",
        help='extract weights from file')
    # parser.add_argument(
    #     '--checkpoint-file',
    #     default='',
    #     help='load checkpoint file')
    # parser.add_argument(
    #     '--ignore-weights',
    #     type=str,
    #     default=[],
    #     nargs='+',
    #     help='the name of weights which will be ignored in the initialization')
    # parser.add_argument(
    #     '--weight2',
    #     default=None,
    #     help='Weight for model 2')

    # =============================================================================
    # Training Dataset Configuration
    # =============================================================================
    parser.add_argument(
        '--train-dataset',
        help='Training dataset type.',
        default='Pose_300W_LP',
        type=str)
    parser.add_argument(
        '--train-data-path',
        help='Directory path for training data.',
        default='./data/resize_300W_LP',
        type=str)
    parser.add_argument(
        '--train-file-name',
        help='Path to text file containing relative paths for training examples.',
        default='./file_list/protocol_1/300W_LP.txt',
        type=str)

    # =============================================================================
    # Test Dataset Configuration
    # =============================================================================
    # Test Dataset 1
    parser.add_argument(
        '--test-dataset1',
        help='Test dataset 1 type.',
        default='BIWI',
        type=str)
    parser.add_argument(
        '--test-data-path1',
        help='Directory path for test data 1.',
        default='./data/resize_BIWI',
        type=str)
    parser.add_argument(
        '--test-file-name1',
        help='Path to text file containing relative paths for test examples 1.',
        default='./file_list/protocol_1/BIWI.txt',
        type=str)

    # Test Dataset 2
    parser.add_argument(
        '--test-dataset2',
        help='Test dataset 2 type.',
        default='AFLW2000',
        type=str)
    parser.add_argument(
        '--test-data-path2',
        help='Directory path for test data 2.',
        default='./data/resize_AFLW2000',
        type=str)
    parser.add_argument(
        '--test-file-name2',
        help='Path to text file containing relative paths for test examples 2.',
        default='./file_list/protocol_1/AFLW2000.txt',
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
    # parser.add_argument(
    #     '--optimizer',
    #     default='SGD',
    #     help='type of optimizer')
    # parser.add_argument(
    #     '--nesterov',
    #     type=str2bool,
    #     default=False,
    #     help='use nesterov momentum')
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

    # Training behavior
    # parser.add_argument(
    #     '--continue-training',
    #     type=str2bool,
    #     default=False,
    #     help='continue training from checkpoint')
    # parser.add_argument(
    #     '--only-train-part',
    #     type=str2bool,
    #     default=False,
    #     help='only train part of the model')
    # parser.add_argument(
    #     '--only-train-epoch',
    #     type=int,
    #     default=0,
    #     help='only train for specific epochs')

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
    # parser.add_argument(
    #     '--log-interval',
    #     type=int,
    #     default=100,
    #     help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    # parser.add_argument(
    #     '--eval-interval',
    #     type=int,
    #     default=2,
    #     help='the interval for evaluating models (#iteration)')
    # parser.add_argument(
    #     '--show-topk',
    #     type=int,
    #     default=[1, 5],
    #     nargs='+',
    #     help='which Top K accuracy will be shown')

    # =============================================================================
    # Testing Configuration
    # =============================================================================
    # parser.add_argument(
    #     '--test-loop',
    #     type=str2bool,
    #     default=False,
    #     help='test all models in a loop')

    return parser

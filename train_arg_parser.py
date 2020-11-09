import argparse as argparse
from pprint import pprint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', default=None, help="File containing the configuration")
    
    # Dataset parameters
    parser.add_argument('-d','--dataset', default='../data/train_test_folder', help="Directory of the dataset")
    parser.add_argument('--train_folder', default='h_train', help="Train folder name")
    parser.add_argument('--val_folder', default='h_test', help="Validation folder name")
    parser.add_argument('--image_folder', default='images', help="Image folder name")
    parser.add_argument('--label_folder', default='groundtruth', help="Label folder name")
    parser.add_argument('--image_ext', default='png', help="Image extension")
    parser.add_argument('--label_ext', default='gt.txt', help="Label image extension")


    # Image generation parameters
    parser.add_argument('--max_side', type=int, default=1024, help="Max side of an image (should be a multiple of 2)")
    parser.add_argument('--use_rotation', type=bool, default=True, help="Whether or not to use rotations as data augmentation")
    parser.add_argument('--theta_range', type=int, default=360, help="The rotation maximum range [-theta; theta] if use_rotation is enabled")
    parser.add_argument('--use_flip', type=bool, default=False, help="Whether or not to use flip as data augmentation")
    parser.add_argument('--crop_size', type=int, default=128, help="The crop size on left/right and top/bottom for the prediction")

    parser.add_argument('--use_rescaling', type=bool, default=True, help="Wether or not to rescale training images")
    parser.add_argument('--min_scale', type=float, default=0.75, help="Min scale for online rescaling")
    parser.add_argument('--max_scale', type=float, default=1.5, help="Max scale for online rescaling")
    parser.add_argument('--val_scale', type=float, default=1.0, help="Validation scale")
    
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--epochs', type=int, default=250, help="Number of epochs to train for")
    parser.add_argument('--start_epoch', type=int, default=0, help="The epoch to start from")
    parser.add_argument('--max_it_per_epoch', type=int, default=10000, help="Maximum number of iterations per epoch")
    parser.add_argument('--loss', default="ctc", help="The loss function to use.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    
    # Network parameters
    parser.add_argument('-l', '--load_path', default=None, help="Directory for loading a trained model")

    

    args = parser.parse_args()
    
    # if args.config_file is not None:
    #     args = load_config(args)
    

    print("Configuration is the following:\n{}".format(args))
    return check_args(args)

def check_args(args):
    return args

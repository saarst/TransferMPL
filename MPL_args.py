from MPL_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data_dir', default='datasets/flowers', type=str, help='data path')
parser.add_argument('--num_labels_percent', type=int, default=0.03, help='percent of labeled data')

parser.add_argument('--num_epochs', default=25, type=int, help='number of epochs to run')

parser.add_argument('--warmup_epoch_num', default=5, type=int, help='warmup steps')

parser.add_argument('--model_name', default='vgg', type=str, help='model name for feature extracting')

parser.add_argument('--unsupervised', default='CE', type=str, help='loss for unsupervised, can be "CE" or "cos" ')

parser.add_argument('--finetune_epochs', default=25, type=int, help='finetune epochs')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda_steps', default=1, type=float, help='warmup steps of lambda-u')

parser.add_argument('--show_images', default=False,  action='store_true', help='show samples from dataset and sample from augmented dataset')
parser.add_argument('--load_best', default=False,  action='store_true', help='load best model to that dataset')
parser.add_argument('--print_model', default=False,  action='store_true', help='print the model we are training')
parser.add_argument('--optuna_mode', default=False,  action='store_true', help='for running optuna')
parser.add_argument('--test', action='store_true', default=False, help='only test')
parser.add_argument('--switch_mode', action='store_true', default=False, help='switch models every epoch')

parser.add_argument('--finetune', action='store_true', default=False,
                    help='only finetune model on labeled dataset')

parser.add_argument('--n_trials', default=10000, type=int, help='n_trials for optuna')
parser.add_argument('--timeout', default=10800, type=int, help='timeout [sec] for optuna')

def add_args(args):
    args.batch_size = 2 if args.data_dir == 'datasets/hymenoptera_data' else 64
    args.num_workers = 2 if torch.cuda.is_available() else 0
    args.pin_memory = True if torch.cuda.is_available() else False
    args.val_size_percentage = 0.008
    args.test_size_percentage = 0.06
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
    args.results_dir = os.path.join('.', 'results', args.name + "_" + args.datetime)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    print(args.name + "  " + args.datetime)
    return args

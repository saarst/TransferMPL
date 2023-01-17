import matplotlib.pyplot as plt
import torch.optim

from MPL_utils import *
from MPL_Data import *
from MPL_visualization import *

#args:

args.batch_size = 64
# args.data_dir = 'datasets/hymenoptera_data'
args.seed = 1
args.data_dir = 'datasets/flowers'
args.val_size_percentage = 0.008
args.test_size_percentage = 0.2
args.num_workers = 2 if torch.cuda.is_available() else 0
args.pin_memory = True if torch.cuda.is_available() else False
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
args.num_labels_percent = 0.04
args.num_epochs = 20            # Number of epochs to train for
args.model_name = "vgg"         # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet]
args.feature_extract = True     # Flag for feature extracting. When False, we fine-tune the whole model,  when True we only update the reshaped layer params
args.temperature = 1
args.threshold = 0
args.lambda_u = 1
args.uda_steps = 1
args.warmup_epoch_num = 5
args.unsupervised = "cos"
args.show_images = False
args.load_best = False
args.print_model = False


dataloaders, dataset_sizes = get_loaders(args)
aug = get_aug()
if args.show_images:
    show_images(args , dataloaders['labeled'], aug)

print(device)

# Initialize the model for this run
t_model, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)
s_model, _ = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)


if args.print_model:
    # Print the model we just instantiated
    print(t_model)   # student has same model

t_model = t_model.to(device)
s_model = s_model.to(device)

if args.load_best:
    print('==> Loading best model ...')
    subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1])
    state = torch.load(os.path.join(subdir, 'best_student.pth'), map_location=device)
    s_model.load_state_dict(state['student'])
    t_model.load_state_dict(state['teacher'])

# Gather the parameters to be optimized/updated in this run. If we are
#  fine-tuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
t_params_to_update = extract_params_to_learn(t_model, args.feature_extract)
s_params_to_update = extract_params_to_learn(s_model, args.feature_extract)

# Observe that all parameters are being optimized
# t_optimizer = torch.optim.SGD(t_params_to_update, lr=0.001, momentum=0.9)
# s_optimizer = torch.optim.SGD(s_params_to_update, lr=0.001, momentum=0.9)

t_optimizer = torch.optim.RAdam(t_params_to_update)
s_optimizer = torch.optim.RAdam(s_params_to_update)

t_scheduler = torch.optim.lr_scheduler.OneCycleLR(t_optimizer, max_lr=0.01, steps_per_epoch=dataset_sizes['labeled'], epochs=args.num_epochs)
s_scheduler = torch.optim.lr_scheduler.OneCycleLR(s_optimizer, max_lr=0.01, steps_per_epoch=dataset_sizes['labeled'], epochs=args.num_epochs)


# Setup the loss fn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
#t_model, hist = train_model_labeled_ref(t_model, dataloaders, criterion, t_optimizer, num_epochs=args.num_epochs)

s_model, t_model, hist = train_model_2(args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler, s_optimizer, s_scheduler, aug)

show_graphs(args, hist)
show_confusionMat(args, s_model, dataloaders['test'], "Student")
show_confusionMat(args, t_model, dataloaders['test'], "Teacher")




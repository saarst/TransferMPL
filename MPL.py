import matplotlib.pyplot as plt
import torch.optim

from MPL_utils import *
from MPL_Data import *
from MPL_visualization import *
from MPL_Optuna import *
from MPL_args import *

# args:

debug = True  # disable during debug
if debug:
    sys.argv = sys.argv + ['--name', 'debug',
                           '--num_epochs', '1',
                           '--model_name', 'vgg',
                           '--data_dir', 'datasets/hymenoptera_data',
                           '--finetune_mode']
args = parser.parse_args()
args = add_args(args)


dataloaders, dataset_sizes = get_loaders(args)
aug = get_aug()
if args.show_images:
    show_images(args, dataloaders['labeled'], aug)

print(device)

# Set up the loss fn
criterion = nn.CrossEntropyLoss()


if args.optuna_mode:
    run_optuna(args, criterion, dataloaders, dataset_sizes, aug)
else:

    # Initialize the model for this run
    t_model, input_size = initialize_model(args.model_name, args.num_classes, use_pretrained=True)
    s_model, _ = initialize_model(args.model_name, args.num_classes, use_pretrained=True)

    if args.print_model:
        # Print the model we just instantiated
        print(t_model)   # student has same model

    t_model = t_model.to(device)
    s_model = s_model.to(device)

    if args.load_best:
        print('==> Loading best model ...')
        subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1], args.name)
        state = torch.load(os.path.join(subdir, 'best_student.pth'), map_location=device)
        s_model.load_state_dict(state['student'])
        t_model.load_state_dict(state['teacher'])

    # Gather the parameters to be optimized/updated in this run. If we are
    #  fine-tuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    t_params_to_update = extract_params_to_learn(t_model)
    s_params_to_update = extract_params_to_learn(s_model)

    # Observe that all parameters are being optimized
    # t_optimizer = torch.optim.SGD(t_params_to_update, lr=0.001, momentum=0.9)
    # s_optimizer = torch.optim.SGD(s_params_to_update, lr=0.001, momentum=0.9)

    t_optimizer = torch.optim.AdamW(t_params_to_update, weight_decay=0.097)
    s_optimizer = torch.optim.AdamW(s_params_to_update, weight_decay=0.097)

    t_scheduler = torch.optim.lr_scheduler.OneCycleLR(t_optimizer, max_lr=0.00263, steps_per_epoch=ceil(dataset_sizes['labeled'] / args.batch_size), epochs=args.num_epochs)
    s_scheduler = torch.optim.lr_scheduler.OneCycleLR(s_optimizer, max_lr=0.00263, steps_per_epoch=ceil(dataset_sizes['labeled'] / args.batch_size), epochs=args.num_epochs)


    # Train and evaluate
    #t_model, hist = train_model_labeled_ref(t_model, dataloaders, criterion, t_optimizer, num_epochs=args.num_epochs)
    if args.switch_mode:
        model_2, model_1, hist = train_model_switch(None, args, t_model, s_model, dataloaders, criterion, t_optimizer,
                                                    t_scheduler, s_optimizer, s_scheduler, aug)
        show_graphs_switcher(args, hist)
        show_confusionMat(args, [model_2], dataloaders['test'], "Model 2")
        show_confusionMat(args, [model_1], dataloaders['test'], "Model 1")
        show_confusionMat(args, [model_1, model_2], dataloaders['test'], "Both models")
    elif args.finetune_mode:
        s_model, hist = train_model_labeled(args, s_model, dataloaders, criterion, s_optimizer, s_scheduler, aug)
        show_graph_1_model(args, hist)
        show_confusionMat(args, [s_model], dataloaders['test'], "Student")
    else:
        s_model, t_model, hist = train_model(None, args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler,
                                             s_optimizer, s_scheduler, aug)
        show_graphs(args, hist)
        show_confusionMat(args, [s_model], dataloaders['test'], "Student")
        show_confusionMat(args, [t_model], dataloaders['test'], "Teacher")
        show_confusionMat(args, [t_model, s_model], dataloaders['test'], "Teacher and Student")

    with open(os.path.join(args.results_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

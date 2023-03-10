from utils import *
from models import *
import torch.optim as optim


def objective(trial, args, criterion, dataloaders, dataset_sizes, aug):
    t_model, input_size = initialize_model(args.model_name, args.num_classes, use_pretrained=True)
    s_model, _ = initialize_model(args.model_name, args.num_classes, use_pretrained=True)
    t_model = t_model.to(device)
    s_model = s_model.to(device)
    t_params_to_update = extract_params_to_learn(t_model)
    s_params_to_update = extract_params_to_learn(s_model)

    weight_decay = trial.suggest_float("weight_decay", 0, 0.1)

    t_optimizer = optim.AdamW(params=t_params_to_update, weight_decay=weight_decay)
    s_optimizer = optim.AdamW(params=s_params_to_update, weight_decay=weight_decay)

    steps_per_epoch = ceil(dataset_sizes['labeled'] / args.batch_size)
    max_steps = steps_per_epoch * args.num_epochs
    max_lr = trial.suggest_float("max_lr", 1e-4, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    t_scheduler = torch.optim.lr_scheduler.OneCycleLR(t_optimizer, max_lr=max_lr,
                                                      steps_per_epoch=steps_per_epoch,
                                                      epochs=args.num_epochs)
    s_scheduler = torch.optim.lr_scheduler.OneCycleLR(s_optimizer, max_lr=max_lr,
                                                      steps_per_epoch=steps_per_epoch,
                                                      epochs=args.num_epochs)
    args.temperature = trial.suggest_float("temperature", 0.5, 3)
    args.lambda_u = trial.suggest_float("lambda_u", 0.1, 2)
    args.uda_steps = trial.suggest_float("uda_steps", 1, 0.2 * max_steps)
    args.threshold = trial.suggest_float("threshold", 0.65, 0.95, step=0.1)

    # Train and evaluate
    _, _, hist = train_model(trial, args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler,
                                         s_optimizer, s_scheduler, aug)
    if len(trial.study.best_trials) > 0:
        with open(os.path.join(args.results_dir, 'optuna.txt'), 'w') as f:
            json.dump(trial.study.best_trial.params, f, indent=2)

    return hist['s_val_acc'][-1]

def run_optuna(args, criterion, dataloaders, dataset_sizes, aug):
    # now we can run the experiment
    print("Starting Optuna:")
    sampler = optuna.samplers.TPESampler()
    func = lambda trial: objective(trial, args,criterion,dataloaders, dataset_sizes, aug)
    study = optuna.create_study(study_name="MPL_Optuna", direction="maximize", sampler=sampler)
    study.optimize(func, args.n_trials, args.timeout)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    with open(os.path.join(args.results_dir, 'optuna.txt'), 'w') as f:
        json.dump(trial.params, f, indent=2)

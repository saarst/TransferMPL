from MPL_utils import *
import torch.optim as optim

def objective(trial, args, t_model,s_model,criterion,dataloaders, dataset_sizes,aug):

    t_params_to_update = extract_params_to_learn(t_model, args.feature_extract)
    s_params_to_update = extract_params_to_learn(s_model, args.feature_extract)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RAdam", "AdamW"])
    t_optimizer = getattr(optim, optimizer_name)(params=t_params_to_update)
    s_optimizer = getattr(optim, optimizer_name)(params=s_params_to_update)

    max_lr = trial.suggest_float("max_lr", 1e-4, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    t_scheduler = torch.optim.lr_scheduler.OneCycleLR(t_optimizer, max_lr=max_lr,
                                                      steps_per_epoch=round(dataset_sizes['labeled'] / args.batch_size),
                                                      epochs=args.num_epochs)
    s_scheduler = torch.optim.lr_scheduler.OneCycleLR(s_optimizer, max_lr=max_lr,
                                                      steps_per_epoch=round(dataset_sizes['labeled'] / args.batch_size),
                                                      epochs=args.num_epochs)

    # Train and evaluate
    s_model, t_model, hist = train_model_2(trial, args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler,
                                           s_optimizer, s_scheduler, aug)

    return hist['s_val_acc'][-1]

def run_optuna(args, t_model, s_model, criterion, dataloaders, dataset_sizes, aug):
    # now we can run the experiment
    sampler = optuna.samplers.TPESampler()
    func = lambda trial: objective(trial, args, t_model,s_model,criterion,dataloaders, dataset_sizes, aug)
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
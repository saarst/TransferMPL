from MPL_models import *


def train_model_labeled(args, model, dataloaders, criterion, optimizer, scheduler, aug):
    since = time.time()
    scaler = torch.cuda.amp.GradScaler()
    aug_weak = aug['aug_weak']

    train_loss_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        # train:
        start_of_train = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data:
        for inputs_l, labels in dataloaders['labeled']:

            inputs_l = inputs_l.to(device)
            inputs_l = aug_weak(inputs_l)
            labels = labels.to(device)

            # teacher: calc t_loss_labels(supervised) and t_loss_uda(unsupervised)
            # forward:
            logits_l = model(inputs_l)
            # Loss:
            loss = criterion(logits_l, labels.long())  # supervised

            # teacher Backward:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # statistics
            running_loss += loss.item() * inputs_l.size(0)

        train_loss_history.append(running_loss / len(dataloaders['labeled'].dataset))

        epoch_train_time = time.time() - start_of_train

        print('Train: Loss: {:.4f}, Epoch train time: {:.0f}m {:.0f}s'.format(
            train_loss_history[-1], epoch_train_time // 60, epoch_train_time % 60))


        # Validation Phase:
        start_of_valid = time.time()
        model.eval()  # Set s_model to evaluate mode
        running_corrects = 0.0
        for inputs_l, labels in dataloaders['val']:
            inputs_l = inputs_l.to(device)
            labels = labels.to(device)

            outputs = model(inputs_l)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        val_acc_history.append(running_corrects.double().cpu() / len(dataloaders['val'].dataset))

        epoch_valid_time = time.time() - start_of_valid
        print('Val: Acc: {:.4f}, Epoch validation time: {:.0f}m {:.0f}s'.format(
            val_acc_history[-1], epoch_valid_time // 60, epoch_valid_time % 60))

        if val_acc_history[-1] > best_acc:
            best_s_acc = val_acc_history[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'args': args
            }
            subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1], args.name)
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            print('==> Saving model ...')
            torch.save(state, os.path.join(subdir, 'best_student.pth'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {"s_val_acc": val_acc_history, "s_train_loss": train_loss_history}



def train_model(trial, args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler, s_optimizer, s_scheduler, aug):
    since = time.time()

    t_scaler = torch.cuda.amp.GradScaler()
    s_scaler = torch.cuda.amp.GradScaler()

    aug_weak = aug['aug_weak']
    aug_strong = aug['aug_strong']

    s_train_loss_history = []
    t_train_loss_history = []

    s_val_acc_history = []
    t_val_acc_history = []

    best_t_model_wts = copy.deepcopy(t_model.state_dict())
    best_s_model_wts = copy.deepcopy(s_model.state_dict())
    best_s_acc = 0.0
    best_t_acc = 0.0

    unlabeled_iter = itertools.cycle(dataloaders['unlabeled'])
    step = -1
    for epoch in range(args.num_epochs):  # this is epochs w.r.t the labeled dataset
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # train:
        start_of_train = time.time()
        t_model.train()  # Set model to training mode
        s_model.train()  # Set model to training mode
        s_running_loss = 0.0
        t_running_loss = 0.0

        # Iterate over data:
        for inputs_l, labels in dataloaders['labeled']:
            step = step + 1
            inputs_l = inputs_l.to(device)
            inputs_l = aug_weak(inputs_l)
            labels = labels.to(device)
            inputs_u, _ = next(unlabeled_iter)

            inputs_u = inputs_u.to(device)
            inputs_uw = aug_weak(inputs_u)    # kornia
            inputs_us = aug_strong(inputs_u)  # kornia

            # teacher: calc t_loss_labels(supervised) and t_loss_uda(unsupervised)
            # forward:
            batch_size = inputs_l.shape[0]
            t_inputs = torch.cat((inputs_l, inputs_uw, inputs_us))
            t_logits = t_model(t_inputs)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits
            # Loss:
            t_loss_l = criterion(t_logits_l, labels.long()) # supervised
            soft_pseudo_label_w = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label_w = torch.max(soft_pseudo_label_w, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            if args.unsupervised == "CE":
                t_loss_u = torch.mean(
                    -(soft_pseudo_label_w * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                )
            else:
                t_loss_u = torch.mean(-cos(soft_pseudo_label_w, torch.softmax(t_logits_uw.detach(), dim=-1)))
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            # student part - calc s_loss for student backward step (if finished warmup), and s_loss_l_old for later use of teacher
            # student Forward:
            s_inputs = torch.cat((inputs_l, inputs_us))
            s_logits = s_model(s_inputs)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits
            # Loss:
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), labels.long()) # for later teacher use, "pre - exam"
            s_loss = criterion(s_logits_us, hard_pseudo_label_w.long()) # for student use, train on pseudo labels

            if epoch >= args.warmup_epoch_num:     # if we finished warmup, student can also train (1st part)
                # student Backward:
                s_optimizer.zero_grad()
                s_scaler.scale(s_loss).backward()
                s_scaler.step(s_optimizer)
                s_scaler.update()

                # now calc t_loss_mps
                with torch.no_grad():
                    s_logits_l = s_model(inputs_l)
                _, s_preds = torch.max(s_logits_l, 1)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), labels.long())    # for teacher use, "exam"
                s_diff = s_loss_l_old - s_loss_l_new
                _, hard_pseudo_label_s = torch.max(t_logits_us.detach(), dim=-1)
                # first term is how much the student has learn, second term is how much teacher is confident:
                t_loss_mpl = s_diff * F.cross_entropy(t_logits_us, hard_pseudo_label_s.long())
                t_loss = t_loss_uda + t_loss_mpl

            else:  # in warmup case, student doesn't learn, so teacher's loss is only from itself
                t_loss = t_loss_uda

            # teacher Backward:
            t_optimizer.zero_grad()
            t_scaler.scale(t_loss).backward()
            t_scaler.step(t_optimizer)
            t_scaler.update()

            t_scheduler.step()
            if epoch >= args.warmup_epoch_num:
                s_scheduler.step()

            # statistics
            s_running_loss += s_loss.item() * inputs_l.size(0)
            t_running_loss += t_loss.item() * inputs_l.size(0)


        s_train_loss_history.append(s_running_loss / len(dataloaders['labeled'].dataset))
        t_train_loss_history.append(t_running_loss / len(dataloaders['labeled'].dataset))

        epoch_train_time = time.time() - start_of_train

        if epoch >= args.warmup_epoch_num:
            print('Train: T_Loss: {:.4f} , S_Loss: {:.4f} ,Epoch train time: {:.0f}m {:.0f}s'.format(t_train_loss_history[-1], s_train_loss_history[-1], epoch_train_time // 60, epoch_train_time % 60))
        else:
            # warmup
            print('Train - Warmup: T_Loss: {:.4f} ,Epoch train time: {:.0f}m {:.0f}s'.format(t_train_loss_history[-1], epoch_train_time // 60, epoch_train_time % 60))

        # Validation Phase:
        start_of_valid = time.time()
        s_model.eval()  # Set s_model to evaluate mode
        t_model.eval()
        s_running_corrects = 0.0
        t_running_corrects = 0.0
        for inputs_l, labels in dataloaders['val']:
            inputs_l = inputs_l.to(device)
            labels = labels.to(device)

            outputs_s = s_model(inputs_l)
            _, s_preds = torch.max(outputs_s, 1)
            s_running_corrects += torch.sum(s_preds == labels.data)

            outputs_t = t_model(inputs_l)
            _, t_preds = torch.max(outputs_t, 1)
            t_running_corrects += torch.sum(t_preds == labels.data)

        s_val_acc_history.append(s_running_corrects.double().cpu() / len(dataloaders['val'].dataset))
        t_val_acc_history.append(t_running_corrects.double().cpu() / len(dataloaders['val'].dataset))


        epoch_valid_time = time.time() - start_of_valid
        print('Val: T_Acc: {:.4f} , S_Acc: {:.4f} ,Epoch validation time: {:.0f}m {:.0f}s'.format(t_val_acc_history[-1], s_val_acc_history[-1], epoch_valid_time // 60, epoch_valid_time % 60))

        if s_val_acc_history[-1] > best_s_acc:
            best_s_acc = s_val_acc_history[-1]
            if not args.optuna_mode:
                best_s_model_wts = copy.deepcopy(s_model.state_dict())

                state = {
                    'student': s_model.state_dict(),
                    'teacher': t_model.state_dict(),
                    'epoch': epoch,
                    'args': args
                }
                subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1])
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                print('==> Saving model ...')
                torch.save(state, os.path.join(subdir, 'best_student.pth'))

        if t_val_acc_history[-1] > best_t_acc:
            best_t_acc = t_val_acc_history[-1]
            best_t_model_wts = copy.deepcopy(t_model.state_dict())

        if args.optuna_mode:
            # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
            if epoch >= args.warmup_epoch_num:
                trial.report(s_val_acc_history[-1], epoch)
            else:
                trial.report(t_val_acc_history[-1], epoch)
            # then, Optuna can decide if the trial should be pruned
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()




    # all epochs finished:
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best s_val Acc: {:4f}'.format(best_s_acc))
    print('Best val Acc: {:4f}'.format(best_t_acc))

    # load best model weights
    s_model.load_state_dict(best_s_model_wts)
    t_model.load_state_dict(best_t_model_wts)
    return s_model, t_model, {"s_val_acc": s_val_acc_history, "t_val_acc": t_val_acc_history,
                              "s_train_loss": s_train_loss_history, "t_train_loss": t_train_loss_history}




def train_model_switch(trial, args, model_1, model_2, dataloaders, criterion, optimizer_1, scheduler_1, optimizer_2, scheduler_2, aug):
    since = time.time()

    scaler_1 = torch.cuda.amp.GradScaler()
    scaler_2 = torch.cuda.amp.GradScaler()

    aug_weak = aug['aug_weak']
    aug_strong = aug['aug_strong']

    train_loss_history_1 = []
    train_loss_history_2 = []

    val_acc_history_1 = []
    val_acc_history_2 = []

    best_1_model_wts = copy.deepcopy(model_1.state_dict())
    best_2_model_wts = copy.deepcopy(model_2.state_dict())
    best_1_acc = 0.0
    best_2_acc = 0.0

    unlabeled_iter = itertools.cycle(dataloaders['unlabeled'])
    step = -1

    t_model, s_model = model_1, model_2
    t_optimizer, s_optimizer = optimizer_1, optimizer_2
    t_scheduler, s_scheduler = scheduler_1, scheduler_2
    t_scaler, s_scaler = scaler_1, scaler_1
    map = {"teacher": "1", "student": "2"}


    for epoch in range(args.num_epochs):  # this is epochs w.r.t the labeled dataset
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        # train:
        start_of_train = time.time()
        t_model.train()  # Set model to training mode
        s_model.train()  # Set model to training mode
        s_running_loss = 0.0
        t_running_loss = 0.0

        # Iterate over data:
        for inputs_l, labels in dataloaders['labeled']:
            step = step + 1
            inputs_l = inputs_l.to(device)
            inputs_l = aug_weak(inputs_l)
            labels = labels.to(device)
            inputs_u, _ = next(unlabeled_iter)

            inputs_u = inputs_u.to(device)
            inputs_uw = aug_weak(inputs_u)    # kornia
            inputs_us = aug_strong(inputs_u)  # kornia

            # teacher: calc t_loss_labels(supervised) and t_loss_uda(unsupervised)
            # forward:
            batch_size = inputs_l.shape[0]
            t_inputs = torch.cat((inputs_l, inputs_uw, inputs_us))
            t_logits = t_model(t_inputs)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits
            # Loss:
            t_loss_l = criterion(t_logits_l, labels.long()) # supervised
            soft_pseudo_label_w = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label_w = torch.max(soft_pseudo_label_w, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            if args.unsupervised == "CE":
                t_loss_u = torch.mean(
                    -(soft_pseudo_label_w * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                )
            else:
                t_loss_u = torch.mean(-cos(soft_pseudo_label_w, torch.softmax(t_logits_uw.detach(), dim=-1)))
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            # student part - calc s_loss for student backward step (if finished warmup), and s_loss_l_old for later use of teacher
            # student Forward:
            s_inputs = torch.cat((inputs_l, inputs_us))
            s_logits = s_model(s_inputs)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits
            # Loss:
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), labels.long()) # for later teacher use, "pre - exam"
            s_loss = criterion(s_logits_us, hard_pseudo_label_w.long()) # for student use, train on pseudo labels

            if epoch >= args.warmup_epoch_num:     # if we finished warmup, student can also train (1st part)
                # student Backward:
                s_optimizer.zero_grad()
                s_scaler.scale(s_loss).backward()
                s_scaler.step(s_optimizer)
                s_scaler.update()

                # now calc t_loss_mps
                with torch.no_grad():
                    s_logits_l = s_model(inputs_l)
                _, s_preds = torch.max(s_logits_l, 1)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), labels.long())    # for teacher use, "exam"
                s_diff = s_loss_l_old - s_loss_l_new
                _, hard_pseudo_label_s = torch.max(t_logits_us.detach(), dim=-1)
                # first term is how much the student has learn, second term is how much teacher is confident:
                t_loss_mpl = s_diff * F.cross_entropy(t_logits_us, hard_pseudo_label_s.long())
                t_loss = t_loss_uda + t_loss_mpl

            else:  # in warmup case, student doesn't learn, so teacher's loss is only from itself
                t_loss = t_loss_uda

            # teacher Backward:
            t_optimizer.zero_grad()
            t_scaler.scale(t_loss).backward()
            t_scaler.step(t_optimizer)
            t_scaler.update()

            t_scheduler.step()
            if epoch >= args.warmup_epoch_num:
                s_scheduler.step()



            # statistics
            s_running_loss += s_loss.item() * inputs_l.size(0)
            t_running_loss += t_loss.item() * inputs_l.size(0)

        running_losses_dict = {map["teacher"]: t_running_loss, map["student"]: s_running_loss}
        train_loss_history_1.append(running_losses_dict["1"] / len(dataloaders['labeled'].dataset))
        train_loss_history_2.append(running_losses_dict["2"] / len(dataloaders['labeled'].dataset))

        epoch_train_time = time.time() - start_of_train

        print('Train: 1_Loss: {:.4f} , 2_Loss: {:.4f} ,Epoch train time: {:.0f}m {:.0f}s'.format(train_loss_history_1[-1], train_loss_history_2[-1], epoch_train_time // 60, epoch_train_time % 60))

        # Validation Phase:
        start_of_valid = time.time()
        s_model.eval()  # Set s_model to evaluate mode
        t_model.eval()
        s_running_corrects = 0.0
        t_running_corrects = 0.0
        for inputs_l, labels in dataloaders['val']:
            inputs_l = inputs_l.to(device)
            labels = labels.to(device)

            outputs_s = s_model(inputs_l)
            _, s_preds = torch.max(outputs_s, 1)
            s_running_corrects += torch.sum(s_preds == labels.data)

            outputs_t = t_model(inputs_l)
            _, t_preds = torch.max(outputs_t, 1)
            t_running_corrects += torch.sum(t_preds == labels.data)

        running_corrects_dict = {map["teacher"]: t_running_corrects, map["student"]: s_running_corrects}
        val_acc_history_1.append(running_corrects_dict["1"].double().cpu() / len(dataloaders['val'].dataset))
        val_acc_history_2.append(running_corrects_dict["2"].double().cpu() / len(dataloaders['val'].dataset))


        epoch_valid_time = time.time() - start_of_valid
        print('Val: 1_Acc: {:.4f} , 2_Acc: {:.4f} ,Epoch validation time: {:.0f}m {:.0f}s'.format(val_acc_history_1[-1], val_acc_history_2[-1], epoch_valid_time // 60, epoch_valid_time % 60))

        models_dict = {map["teacher"]: t_model, map["student"]: s_model}
        if val_acc_history_2[-1] > best_2_acc:
            best_2_acc = val_acc_history_2[-1]
            best_2_model_wts = copy.deepcopy(models_dict["2"].state_dict())
            print('==> Saving model ...')
            state = {
                '1': models_dict["1"].state_dict(),
                '2': models_dict["2"].state_dict(),
                'epoch': epoch,
                'args': args
            }
            subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1])
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            torch.save(state, os.path.join(subdir, 'best_switcher.pth'))

        if val_acc_history_1[-1] > best_1_acc:
            best_1_acc = val_acc_history_1[-1]
            best_1_model_wts = copy.deepcopy(models_dict["1"].state_dict())

        if args.optuna_mode:
            # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
            trial.report(val_acc_history_2[-1], epoch)
            # then, Optuna can decide if the trial should be pruned
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        # switch models:
        t_model, s_model = s_model, t_model
        t_optimizer, s_optimizer = s_optimizer, t_optimizer
        t_scheduler, s_scheduler = s_scheduler, t_scheduler
        t_scaler, s_scaler = s_scaler, t_scaler
        new_map = {"teacher": map["student"], "student": map["teacher"]}
        map = new_map

    models_dict = {map["teacher"]: t_model, map["student"]: s_model}

    # all epochs finished:
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best 1_val Acc: {:4f}'.format(best_1_acc))
    print('Best 2_val Acc: {:4f}'.format(best_2_acc))

    # load best model weights
    models_dict["1"].load_state_dict(best_1_model_wts)
    models_dict["2"].load_state_dict(best_2_model_wts)
    return models_dict["2"], models_dict["1"], {"2_val_acc": val_acc_history_2, "1_val_acc": val_acc_history_1,
                              "2_train_loss": train_loss_history_2, "1_train_loss": train_loss_history_1}
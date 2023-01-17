# imports for the tutorial
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import itertools
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
import torchvision
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
import pandas as pd
import seaborn as sn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
varImageNet = [0.229, 0.224, 0.225]
invVar = [1/0.229, 1/0.224, 1/0.225]
meanImageNet = [0.485, 0.456, 0.406]
invMean = [-0.485,- 0.456, -0.406]




def set_parameter_requires_grad(model, feature_extracting=False):
    # approach 1
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        model.requires_grad_(True)

    # approach 2
    if feature_extracting:
        # frozen model
        for param in model.parameters():
            param.requires_grad = False
    else:
        # fine-tuning
        for param in model.parameters():
            param.requires_grad = True
    # note: you can also mix between frozen layers and trainable layers, but you'll need a custom
    # function that loops over the model's layers and you specify which layers are frozen.


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0  # image size, e.g. (3, 224, 224)
    # new method from torchvision >= 0.13
    weights = 'DEFAULT' if use_pretrained else None
    # to use other checkpoints than the default ones, check the model's available chekpoints here:
    # https://pytorch.org/vision/stable/models.html
    if model_name == "resnet":
        """ Resnet18
        """
        # new method from torchvision >= 0.13
        model_ft = models.resnet18(weights=weights)
        # old method for toechvision < 0.13
        # model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        # new method from torchvision >= 0.13
        model_ft = models.alexnet(weights=weights)
        # old method for toechvision < 0.13
        # model_ft = models.alexnet(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16
        """
        # new method from torchvision >= 0.13
        model_ft = models.vgg16(weights=weights)
        # old method for toechvision < 0.13
        # model_ft = models.vgg16(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        # new method from torchvision >= 0.13
        model_ft = models.squeezenet1_0(weights=weights)
        # old method for toechvision < 0.13
        # model_ft = models.squeezenet1_0(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        # new method from torchvision >= 0.13
        model_ft = models.densenet121(weights=weights)
        # old method for toechvision < 0.13
        # model_ft = models.densenet121(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        raise NotImplementedError

    return model_ft, input_size


def extract_params_to_learn(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update


"""
Training function
"""


def train_model_labeled_ref(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    s_val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['labeled', 'val']:
            if phase == 'labeled':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'labeled'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'labeled':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


class args:
    x = 1


def x_split(args, labels, size):
    label_per_class = size // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == label_per_class * args.num_classes
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def x_split_separate(args, labels, size, labels_indexes=None, separate=True):
    if labels_indexes is None:
        labels_indexes = np.array(range(len(labels)))
    label_per_class = size // args.num_classes
    labels = np.array(labels)
    test_idx = []
    train_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        test_idx.extend(idx[:label_per_class])
        train_idx.extend(idx[label_per_class:])
    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    assert len(test_idx) == label_per_class * args.num_classes
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    if separate:
        return labels_indexes[test_idx], labels_indexes[train_idx]
    else:
        return labels_indexes[test_idx], labels_indexes


def check_idx(train_idx, val_idx, test_idx, total_len):
    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    test_idx = test_idx.tolist()
    all_idx = np.array(range(total_len)).tolist()

    sets = [set(train_idx), set(val_idx), set(test_idx), set(all_idx)]
    empty_set = set()
    a = sets[0] & sets[1]
    b = sets[0] & sets[2]
    c = sets[1] & sets[2]

    d = sets[0] | sets[1] | sets[2]

    assert(a == empty_set)
    assert(b == empty_set)
    assert(c == empty_set)
    assert(d == sets[3])
    return







def train_model(args, t_model, s_model, dataloaders, criterion, t_optimizer, s_optimizer):
    since = time.time()
    cos = nn.CosineSimilarity()
    aug_weak = AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.Normalize(meanImageNet, varImageNet),
        same_on_batch=False,
    )
    aug_strong = AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.Normalize(meanImageNet, varImageNet),
        K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.2),
        K.RandomAffine((-15., 20.), (0.1, 0.1), (0.7, 1.2), (30., 50.), p=0.3),
        K.RandomPerspective(0.5, p=0.3),
        K.RandomGrayscale(p=0.1),
        K.RandomGaussianNoise(0, 0.1, p=0.2),
        same_on_batch=False,
    )
    s_val_acc_history = []
    s_val_loss_history = []

    s_train_loss_history = []
    t_train_loss_history = []

    best_t_model_wts = copy.deepcopy(t_model.state_dict())
    best_s_model_wts = copy.deepcopy(s_model.state_dict())
    best_acc = 0.0
    unlabeled_iter = iter(dataloaders['unlabeled'])
    step = -1
    for epoch in range(args.num_epochs):  # this is epochs w.r.t the labeled dataset
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                dataset_to_iter = 'labeled'  # in the train phase, the epochs are of the labeled set
                t_model.train()  # Set model to training mode
                s_model.train()
            else:
                dataset_to_iter = 'val'
                s_model.eval()  # Set s_model to evaluate mode
                t_model.eval()

            s_running_loss = 0.0
            t_running_loss = 0.0
            s_running_corrects = 0.0
            t_running_corrects = 0.0

            # Iterate over data.
            for inputs_l, labels in dataloaders[dataset_to_iter]:
                inputs_l = inputs_l.to(device)
                inputs_l = aug_weak(inputs_l)
                labels = labels.to(device)

                if phase == 'train':
                    step = step + 1
                    inputs_u, _ = next(unlabeled_iter)
                    inputs_u = inputs_u.to(device)
                    inputs_uw = aug_weak(inputs_u)  # inputs_uw = aug_list_weak(inputs_u)   - kornia
                    inputs_us = aug_strong(inputs_u)  # inputs_us = aug_list_strong(inputs_u) - kornia

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if phase == 'val':
                        outputs = s_model(inputs_l)
                        s_loss = criterion(outputs, labels)
                        _, s_preds = torch.max(outputs, 1)

                        outputs = t_model(inputs_l)
                        t_loss = criterion(outputs, labels)
                        _, t_preds = torch.max(outputs, 1)
                    else:
                        # first teacher part, calc t_loss_labels and t_loss_uda
                        batch_size = inputs_l.shape[0]
                        t_inputs = torch.cat((inputs_l, inputs_uw, inputs_us))
                        t_logits = t_model(t_inputs)
                        t_logits_l = t_logits[:batch_size]
                        t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                        del t_logits
                        t_loss_l = criterion(t_logits_l, labels.long())
                        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
                        max_probs, hard_pseudo_label_on_w = torch.max(soft_pseudo_label, dim=-1)
                        mask = max_probs.ge(args.threshold).float()
                        if args.unsupervised == "CE":
                            t_loss_u = torch.mean(
                                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                            )
                        else:
                            t_loss_u = torch.mean(-cos(soft_pseudo_label, torch.softmax(t_logits_uw.detach(), dim=-1)))

                        weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
                        t_loss_uda = t_loss_l + weight_u * t_loss_u

                        if epoch >= args.warmup_epoch_num:
                            # student part - calc s_loss for student backward step, and s_loss_l_old for later use of teacher
                            s_inputs = torch.cat((inputs_l, inputs_us))
                            s_logits = s_model(s_inputs)
                            s_logits_l = s_logits[:batch_size]
                            s_logits_us = s_logits[batch_size:]
                            del s_logits
                            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), labels.long())
                            s_loss = criterion(s_logits_us, hard_pseudo_label_on_w.long())
                            s_optimizer.zero_grad()
                            s_loss.backward()
                            s_optimizer.step()

                            # now calc t_loss_mps
                            with torch.no_grad():
                                s_logits_l = s_model(inputs_l)
                                _, s_preds = torch.max(s_logits_l, 1)
                            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), labels.long())
                            s_diff = s_loss_l_old - s_loss_l_new
                            _, hard_pseudo_label_on_s = torch.max(t_logits_us.detach(), dim=-1)
                            t_loss_mpl = s_diff * F.cross_entropy(t_logits_us, hard_pseudo_label_on_s.long())
                            t_loss = t_loss_uda + t_loss_mpl

                        else:  # in warmup case, student doesn't learn, so teacher's loss is only from itself
                            t_loss = t_loss_uda
                            with torch.no_grad():
                                s_outputs = s_model(inputs_l)
                                s_loss = F.cross_entropy(s_outputs.detach(), labels.long())
                                _, s_preds = torch.max(s_outputs, 1)

                        t_optimizer.zero_grad()
                        t_loss.backward()
                        t_optimizer.step()

                        with torch.no_grad():
                            outputs = t_model(inputs_l)
                            _, t_preds = torch.max(outputs, 1)

                # statistics
                s_running_loss += s_loss.item() * inputs_l.size(0)
                s_running_corrects += torch.sum(s_preds == labels.data)
                t_running_loss += t_loss.item() * inputs_l.size(0)
                t_running_corrects += torch.sum(t_preds == labels.data)

            s_epoch_loss = s_running_loss / len(dataloaders[dataset_to_iter].dataset)
            s_epoch_acc = s_running_corrects.double().cpu() / len(dataloaders[dataset_to_iter].dataset)
            t_epoch_loss = t_running_loss / len(dataloaders[dataset_to_iter].dataset)
            t_epoch_acc = t_running_corrects.double().cpu() / len(dataloaders[dataset_to_iter].dataset)

            if epoch >= args.warmup_epoch_num or phase == 'val':
                print('{} T_Loss: {:.4f} T_Acc: {:.4f} S_Loss: {:.4f} S_Acc: {:.4f}'.format(phase, t_epoch_loss,
                                                                                            t_epoch_acc, s_epoch_loss,
                                                                                            s_epoch_acc))
            else:
                # warmup
                print('{} : Warmup. T_Loss: {:.4f} T_Acc: {:.4f} '.format(phase, t_epoch_loss, t_epoch_acc))

            # deep copy the model
            if phase == 'val' and s_epoch_acc > best_acc:
                best_acc = s_epoch_acc
                best_s_model_wts = copy.deepcopy(s_model.state_dict())
                best_t_model_wts = copy.deepcopy(t_model.state_dict())
            if phase == 'val':
                s_val_acc_history.append(s_epoch_acc)
                s_val_loss_history.append(s_epoch_loss)
            else:
                s_train_loss_history.append(s_epoch_loss)
                t_train_loss_history.append(t_epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    s_model.load_state_dict(best_s_model_wts)
    t_model.load_state_dict(best_t_model_wts)
    return s_model, t_model, {"s_val_acc": s_val_acc_history, "s_val_loss": s_val_loss_history,
                              "s_train_loss": s_train_loss_history, "t_train_loss": t_train_loss_history}


def train_model_2(args, t_model, s_model, dataloaders, criterion, t_optimizer, t_scheduler, s_optimizer, s_scheduler, aug):
    since = time.time()
    cos = nn.CosineSimilarity()

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
            best_s_model_wts = copy.deepcopy(s_model.state_dict())
            print('==> Saving model ...')
            state = {
                'student': s_model.state_dict(),
                'teacher': t_model.state_dict(),
                'epoch': epoch,
                'args': args
            }
            subdir = os.path.join('.', 'checkpoints', args.data_dir.split("/")[1])
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            torch.save(state, os.path.join(subdir, 'best_student.pth'))

        if t_val_acc_history[-1] > best_t_acc:
            best_t_acc = t_val_acc_history[-1]
            best_t_model_wts = copy.deepcopy(t_model.state_dict())

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


# function to calcualte accuracy of the model
def calculate_accuracy(args, model, dataloader):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([args.num_classes,args.num_classes], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

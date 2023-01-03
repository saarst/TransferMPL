# imports for the tutorial
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
varAntsBees  = [0.229, 0.224, 0.225]
meanAntsBees = [0.485, 0.456, 0.406]
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
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update



"""
Training function
"""
def train_model_labeled_ref(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

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
                model.eval()   # Set model to evaluate mode

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
    x=1


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def train_model(args, t_model, s_model , dataloaders, criterion, t_optimizer,s_optimizer):
    since = time.time()
    aug_weak = AugmentationSequential(
                                    K.RandomHorizontalFlip(),
                                    K.Normalize(meanAntsBees,varAntsBees),

                                    same_on_batch=False,
                                )
    aug_strong = AugmentationSequential(
                                    K.RandomHorizontalFlip(),
                                    K.Normalize(meanAntsBees, varAntsBees),
                                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.1),
                                    K.RandomAffine((-15., 20.), (0.1, 0.1), (0.7, 1.2), (30., 50.), p=0.1),
                                    K.RandomPerspective(0.5, p=0.1),
                                    K.RandomGrayscale(p=0.2),
                                    K.RandomGaussianNoise(0, 0.1, p=0.2),
                                    same_on_batch=False,
                                )
    val_acc_history = []

    best_model_wts = copy.deepcopy(t_model.state_dict())
    best_acc = 0.0
    unlabeled_iter = iter(dataloaders['unlabeled'])
    step = -1
    for epoch in range(args.num_epochs):    # this is epochs w.r.t the labeled dataset
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
                t_model.eval()   # Set model to evaluate mode


            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs_l, labels in dataloaders[dataset_to_iter]:
                inputs_l = inputs_l.to(device)
                inputs_l = aug_weak(inputs_l)
                labels = labels.to(device)

                if phase == 'train':
                    step = step + 1
                    inputs_u, _ = next(unlabeled_iter)
                    inputs_u = inputs_u.to(device)
                    inputs_uw = aug_weak(inputs_u)      # inputs_uw = aug_list_weak(inputs_u)   - kornia
                    inputs_us = aug_strong(inputs_u)    # inputs_us = aug_list_strong(inputs_u) - kornia



                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if phase == 'val':
                        outputs = t_model(inputs_l)
                        t_loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
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
                        t_loss_u = torch.mean(
                            -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                        )
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
                            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), labels.long())
                            s_diff = torch.abs(s_loss_l_old - s_loss_l_new)
                            _, hard_pseudo_label_on_s = torch.max(t_logits_us.detach(), dim=-1)
                            t_loss_mpl = s_diff * F.cross_entropy(t_logits_us, hard_pseudo_label_on_s.long())
                            t_loss = t_loss_uda + t_loss_mpl

                        else: # in warmup case, student doesn't learn, so teacher's loss is only from itself
                            t_loss = t_loss_uda

                        t_optimizer.zero_grad()
                        t_loss.backward()
                        t_optimizer.step()

                        with torch.no_grad():
                            outputs = t_model(inputs_l)
                        _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += t_loss.item() * inputs_l.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[dataset_to_iter].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[dataset_to_iter].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(t_model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    t_model.load_state_dict(best_model_wts)
    return t_model, val_acc_history

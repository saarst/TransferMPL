import logging
import math
import os

from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
import pickle
import torch
import torchvision
from tqdm import tqdm


import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms, models

from augmentation import RandAugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, model):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, model, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


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

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, model, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default
        self.model = model

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.normalize(self.ori(x))
        aug = self.normalize(self.aug(x))
        ori_features = self.model(ori.unsqueeze(0))
        aug_features = self.model(aug.unsqueeze(0))
        return ori_features, aug_features


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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

def initialize_model(model_name, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0  # image size, e.g. (3, 224, 224)
    # new method from torchvision >= 0.13
    weights = 'DEFAULT' if use_pretrained else None
    # to use other checkpoints than the default ones, check the model's available chekpoints here:
    # https://pytorch.org/vision/stable/models.html

    if model_name == "vgg":
        """ VGG16
        """
        # new method from torchvision >= 0.13
        model_ft = models.vgg16(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Identity()
        input_size = 224

    else:
        raise NotImplementedError

    return model_ft, input_size

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}

def features():

    file2 = open(r'./d.pkl', 'rb')
    args = pickle.load(file2)
    file2.close()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.amp = True if torch.cuda.is_available() else False
    args.label_smoothing = 0
    args.workers = 4 if torch.cuda.is_available() else 0
    args.resize = 224
    print(f'pytorch: {torch.__version__}, torchvision: {torchvision.__version__}')

    feature_extract = True
    model_name = "vgg"
    model_ft, input_size = initialize_model(model_name, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(args.device)

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, model_ft)
    print(model_ft)


    os.makedirs('./content/labeled_dataset', exist_ok=True)
    os.makedirs('./content/unlabeled_dataset', exist_ok=True)
    os.makedirs('./content/test_dataset', exist_ok=True)

    os.makedirs('./content/unlabeled_dataset/weak_aug', exist_ok=True)
    os.makedirs('./content/unlabeled_dataset/strong_aug', exist_ok=True)
    os.makedirs('./content/unlabeled_dataset/target', exist_ok=True)


    os.makedirs('./content/labeled_dataset/img', exist_ok=True)
    os.makedirs('./content/labeled_dataset/target', exist_ok=True)

    os.makedirs('./content/test_dataset/img', exist_ok=True)
    os.makedirs('./content/test_dataset/target', exist_ok=True)


    for i, data in tqdm(enumerate(unlabeled_dataset), total=len(unlabeled_dataset)):
        torch.save(data[0][0], './content/unlabeled_dataset/weak_aug/unlabeled_weakAug_img{}'.format(i))
        torch.save(data[0][0], './content/unlabeled_dataset/strong_aug/unlabeled_strongAug_img{}'.format(i))
        torch.save(data[1], './content/unlabeled_dataset/target/unlabeled_target{}'.format(i))

    print("Finish")







import matplotlib.pyplot as plt

from MPL_utils import *


def get_loaders(args):
    if args.data_dir == 'datasets/hymenoptera_data':
        basic_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()])
    elif args.data_dir == 'datasets/flowers':
        basic_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor()])

    dataset = ImageFolder(args.data_dir, basic_transform)
    args.class_names = dataset.classes
    args.num_classes = len(args.class_names)
    targets = np.array(dataset.targets)
    args.test_size = round(args.test_size_percentage * len(dataset))

    test_idx, train_idx = x_split_separate(args, targets, args.test_size, separate=True)

    args.val_size = round(args.val_size_percentage * len(train_idx))
    val_idx, train_idx = x_split_separate(args, targets[train_idx], args.val_size, train_idx, separate=True)

    args.num_labeled = round(args.num_labels_percent * len(train_idx))
    labeled_idx, unlabeled_idx = x_split_separate(args, targets[train_idx], args.num_labeled, train_idx,
                                                  separate=False)

    image_datasets = {'unlabeled': Subset(dataset, unlabeled_idx), 'labeled': Subset(dataset, labeled_idx),
                      'val': Subset(dataset, val_idx), 'test': Subset(dataset, test_idx)}

    check_idx(train_idx, val_idx, test_idx, len(targets))

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory) for x in
                   ['unlabeled', 'labeled', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['unlabeled', 'labeled', 'val', 'test']}

    return dataloaders, dataset_sizes


def get_aug():
    aug_weak = AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.Normalize(meanAntsBees, varAntsBees),
        same_on_batch=False,
    )
    aug_strong = AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.Normalize(meanAntsBees, varAntsBees),
        K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.2),
        K.RandomAffine((-15., 20.), (0.1, 0.1), (0.7, 1.2), (30., 50.), p=0.3),
        K.RandomPerspective(0.5, p=0.3),
        K.RandomGrayscale(p=0.1),
        K.RandomGaussianNoise(0, 0.1, p=0.2),
        same_on_batch=False,
    )
    return {'aug_weak': aug_weak, 'aug_strong': aug_strong}





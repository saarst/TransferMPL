import matplotlib.pyplot as plt

from MPL_utils import *


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


def get_loaders(args):
    basic_transform = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
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
    print(dataset_sizes)

    return dataloaders, dataset_sizes


def get_aug():
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
    return {'aug_weak': aug_weak, 'aug_strong': aug_strong}





import matplotlib.pyplot as plt
import pandas as pd

from MPL_utils import *

def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range - approximately...
    norm = transforms.Normalize(invMean, invVar)
    image = norm(image)
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    image = image.to("cpu").numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


def show_images(args,datalodaer, aug):
    images, labels = next(iter(datalodaer))
    # first no augs:
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for idx, image in enumerate(images):
        if idx == 5:
            break
        axes[idx].imshow(image.permute(1, 2, 0))
        axes[idx].set_title(args.class_names[labels[idx]])
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    fig.tight_layout()
    plt.show()

    # second with augs:
    images = images.to(device)
    aug_strong = aug['aug_strong']
    images_aug = aug_strong(images)

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for idx, image in enumerate(images_aug):
        if idx == 5:
            break
        axes[idx].imshow(convert_to_imshow_format(image))
        axes[idx].set_title(args.class_names[labels[idx]] + "Augmented")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    fig.tight_layout()
    plt.show()


def show_graphs(args, hist):
    x = np.arange(1, args.num_epochs + 1)
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss - student and teacher')
    ax.plot(x, hist['t_train_loss'], x, hist['s_train_loss'])
    ax.legend(["Teacher train loss", "Student train loss"])

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Acc')
    ax.set_title('Validation accuracy - student and teacher')
    ax.plot(x, hist['t_val_acc'], x, hist['s_val_acc'])
    ax.legend(["Teacher val acc", "Student val acc"])

    fig.tight_layout()
    plt.show()


def show_confusionMat(args , model, dataloader,s_or_t):
    test_accuracy, confusion_matrix = calculate_accuracy(args, model, dataloader)
    print(s_or_t + " test accuracy: {:.3f}%".format(test_accuracy))
    df_cm = pd.DataFrame(confusion_matrix, args.class_names, args.class_names)
    # plot confusion matrix
    pp_matrix(df_cm, cmap="PuRd")

from utils import *



def set_parameter_requires_grad(model):
    model.requires_grad_(False)


def initialize_model(model_name, num_classes, use_pretrained=True):
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

        set_parameter_requires_grad(model_ft)
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

        set_parameter_requires_grad(model_ft)
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

        set_parameter_requires_grad(model_ft)
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

        set_parameter_requires_grad(model_ft)
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

        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        raise NotImplementedError

    return model_ft, input_size


def extract_params_to_learn(model_ft):
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            # print("\t", name)
            params_to_update.append(param)
    return params_to_update

from MPL_utils import *

# scikit-learn imports

# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'unlabeled': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#args:

args.batch_size = 4
args.data_dir = 'datasets/hymenoptera_data'
args.num_workers = 4 if torch.cuda.is_available() else 0
args.num_labels_percent = 0.01
args.num_classes = 2
args.num_epochs = 25         # Number of epochs to train for
args.model_name = "vgg"         # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet]
args.feature_extract = True     # Flag for feature extracting. When False, we fine-tune the whole model,  when True we only update the reshaped layer params
args.temperature = 1
args.threshold = 0
args.mask = 0
args.lambda_u = 0
args.uda_steps = 1
args.warmup_epoch_num = 5


basic_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])

#code:
dirs = {'unlabeled': 'train', 'val': 'val'}
image_datasets = {x: ImageFolder(os.path.join(args.data_dir, dirs[x]), basic_transform) for x in ['unlabeled', 'val']}

args.num_labeled = round(args.num_labels_percent * len(image_datasets['unlabeled']))
labeled_idx, _ = x_u_split(args, image_datasets['unlabeled'].targets)
image_datasets['labeled'] = Subset(image_datasets['unlabeled'], labeled_idx)

dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers) for x in ['unlabeled', 'labeled', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['unlabeled', 'labeled', 'val']}
class_names = image_datasets['unlabeled'].classes

print(device)
# Get a batch of training data
#unlabeled_set = dataloaders['unlabeled']

# Initialize the model for this run
t_model, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)
s_model, _ = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)


# Print the model we just instantiated
print(t_model)   # student has same model

t_model = t_model.to(device)
s_model = s_model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  fine-tuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
t_params_to_update = extract_params_to_learn(t_model, args.feature_extract)
s_params_to_update = extract_params_to_learn(s_model, args.feature_extract)

# Observe that all parameters are being optimized
t_optimizer = torch.optim.SGD(t_params_to_update, lr=0.001, momentum=0.9)
s_optimizer = torch.optim.SGD(s_params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
#t_model, hist = train_model_labeled_ref(t_model, dataloaders, criterion, t_optimizer, num_epochs=args.num_epochs)

s_model, t_model, hist = train_model(args, t_model, s_model , dataloaders, criterion, t_optimizer,s_optimizer)
print("Plotting")
x = np.arange(1, args.num_epochs + 1)
s_fig = plt.figure(figsize=(8, 8))
ax = s_fig.add_subplot(1, 1, 1)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss\Acc')
ax.set_title('Loss - student')
ax.plot(x, hist['s_val_acc'], x, hist['s_train_loss'])
ax.legend(["Student Val Accuray", "Student Train Loss"])
s_fig.tight_layout()
plt.show()

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(2, 1, 2)
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy - train and test')
# ax.plot(x, test_accuracy, x, train_accuracy)
# ax.legend(["test Accuracy", "train Accuracy"])




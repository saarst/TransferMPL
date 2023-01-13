import matplotlib.pyplot as plt

from MPL_utils import *
from MPL_Data import *

#args:

args.batch_size = 4
args.data_dir = 'datasets/hymenoptera_data_2'
args.seed = 1
# args.data_dir = 'datasets/flowers'
args.val_size_percentage  = 0.2
args.test_size_percentage = 0.2
args.num_workers = 4 if torch.cuda.is_available() else 0
args.pin_memory = True if torch.cuda.is_available() else False
args.num_labels_percent = 0.1
args.num_epochs = 25         # Number of epochs to train for
args.model_name = "vgg"         # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet]
args.feature_extract = True     # Flag for feature extracting. When False, we fine-tune the whole model,  when True we only update the reshaped layer params
args.temperature = 1
args.threshold = 0
args.mask = 0
args.lambda_u = 0.5
args.uda_steps = 1
args.warmup_epoch_num = 5


dataloaders, dataset_sizes = get_loaders(args)

print(device)

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

s_model, t_model, hist = train_model_2(args, t_model, s_model , dataloaders, criterion, t_optimizer,s_optimizer)


x = np.arange(1, args.num_epochs + 1)
fig = plt.figure(figsize=(10, 10))

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

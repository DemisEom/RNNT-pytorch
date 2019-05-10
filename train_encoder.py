import torch
import models.rnnt_model as models
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logger for visualaization tensorboard
logger = Logger('./logs')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# load data
# MNIST dataset for sample
data_set = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=100,
                                          shuffle=True)

# load model
# model = models.ConvNet().to(device)
# model = models.BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

encoder_model = models.Encoder().to(device)
prediction_network_model = models.PredictionNet().to(device)
joint_network_model = models.JointNetwork().to(device)


# Loss and optimizer
criterion = nn.Softmax()
optimizer = torch.optim.Adam(joint_network_model.parameters(), lr=0.00001)

# setting training parameters
data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000



# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    # Forward pass
    encoder_output = encoder_model(data_x)
    prediction_network_output = prediction_network_model(data_y)
    outputs = joint_network_model(encoder_output, prediction_network_output)

    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(step + 1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

        # 3. Log training images (image summary)
        info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

        for tag, images in info.items():
            logger.image_summary(tag, images, step + 1)


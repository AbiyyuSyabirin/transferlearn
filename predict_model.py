import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn

# Define data transforms
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda")

def visualize_model_prediction(model, img_path, class_names):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        class_names = ['goats', 'sheeps']

        ax = plt.subplot(2, 2, 1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        img = img.cpu().data[0].numpy().transpose((1, 2, 0))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
    
    plt.show()

if __name__ == '__main__':
    # Load the model
    data_dir = 'Pytorch/data/dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['val'])
                      for x in ['val']}
    class_names = image_datasets['val'].classes

    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)

    model_conv.load_state_dict(torch.load('models/best_model_params.pt', weights_only=True))

    visualize_model_prediction(
        model_conv,
        img_path='Pytorch/data/dataset/testing/goat.jpg',
        class_names=class_names
    )
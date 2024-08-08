import torch
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom dataset to load images from directory
class ImagePairDataset(Dataset):
    def __init__(self, real_image_path, generated_image_path, transform=None):
        self.real_image_path = real_image_path
        self.generated_image_path = generated_image_path
        self.transform = transform

    def __len__(self):
        return 2  # 두 개의 출력 이미지

    def __getitem__(self, idx):
        if idx == 0:
            image_path = self.real_image_path
        else:
            image_path = self.generated_image_path

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def calculate_fid(real_features, generated_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.atleast_2d(sigma1)
    if sigma2.ndim == 0:
        sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_features(dataset, model, batch_size=1, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images).view(images.size(0), -1)
            features.append(outputs.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

# Function to compute FID between a real image and generated images
def compute_fid(real_image_path, generated_image_path, device='cuda'):
    # Transformation to apply to images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset
    dataset = ImagePairDataset(real_image_path, generated_image_path, transform=transform)

    # Load pre-trained InceptionV3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Get features from the real and generated images
    features = get_features(dataset, inception_model, device=device)

    # Calculate FID
    real_features = features[0:1]
    generated_features = features[1:]
    fid_score = calculate_fid(real_features, generated_features)
    return fid_score
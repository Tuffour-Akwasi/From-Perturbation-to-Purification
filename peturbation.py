import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
import foolbox as fb
from advertorch.attacks import PGDAttack
from torchvision import models
from PIL import Image
import numpy as np

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Clean Test Accuracy: {accuracy:.2f}%")
import torch.nn as nn

def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()

    ori_images = images.data

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
    
def fgsm_attack(model, images, labels, eps=0.3):
    images = images.clone().detach()
    labels = labels.clone().detach()
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_images = images + eps * images.grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    return adv_images


def evaluate_adversarial(model, dataloader, attack_type="pgd"):
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))

        if attack_type == "pgd":
            adv_images = pgd_attack(model, images, labels)  # ← You need this
        elif attack_type == "fgsm":
            adv_images = fgsm_attack(model, images, labels)  # ← Or something similar
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100. * correct / total
    print(f"Adversarial Test Accuracy ({attack_type.upper()}): {accuracy:.2f}%")

def evaluate_denoised(model, test_loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
        # Simulate attack for demo purposes
        adv_images = images + 0.1 * torch.randn_like(images).clamp(0, 1)
        adv_images = adv_images.to(torch.device("cpu"))
        # Apply diffusion-based denoising (placeholder)
        # This is a simplified placeholder for actual diffusion purification
        denoised_images = adv_images.clamp(0, 1)  # Simulated cleaning
        with torch.no_grad():
            outputs = model(denoised_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Denoised Adversarial Accuracy: {accuracy:.2f}%")

def generate_synthetic(prompt):
    return pipeline(prompt).images[0]

# Load Stable Diffusion Model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(torch.device("cpu"))

# Load Pretrained Model for Adversarial Testing
model = models.resnet18(pretrained=True).to(torch.device("cpu"))
model.eval()

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def attack_model(model, image, label, attack_type="pgd"):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    if attack_type == "pgd":
        attack = fb.attacks.LinfPGD()
    elif attack_type == "fgsm":
        attack = fb.attacks.FGSM()
    
    adversarial, _ = attack(fmodel, image, label)
    return adversarial

# Denoising Defense using Diffusion Model
def denoise_with_diffusion(pipeline, adv_image):
    image_tensor = transform(adv_image).unsqueeze(0).to(torch.device("cpu"))
    denoised_image = pipeline(image_tensor)
    return denoised_image

# Training Pipeline
batch_size = 32
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Training Completed")
    evaluate_model(model, test_loader)
    evaluate_adversarial(model, test_loader, attack_type="pgd")
    evaluate_adversarial(model, test_loader, attack_type="fgsm")
    evaluate_denoised(model, test_loader)

train_model(model, train_loader)

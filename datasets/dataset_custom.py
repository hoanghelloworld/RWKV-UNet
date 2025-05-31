import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None, is_test=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image filenames
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_filenames.sort()
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        
        if not self.is_test:
            # Load mask for training/validation
            mask_filename = self.image_filenames[idx].replace('.jpg', '.png')
            mask_path = os.path.join(self.masks_dir, mask_filename)
            mask = Image.open(mask_path).convert('L')
            
            # Convert to numpy arrays
            image = np.array(image)
            mask = np.array(mask)
            
            # Convert mask to binary (0, 1)
            mask = (mask > 127).astype(np.uint8)
            
            # Apply transforms if provided
            if self.transform:
                # Convert to tensor and normalize
                image = transforms.ToTensor()(image)
                image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
                
                # Convert to grayscale for model input (take only one channel)
                image = transforms.Grayscale()(image)
                
                mask = torch.from_numpy(mask).long()
            
            sample = {
                'image': image,
                'label': mask,
                'case_name': self.image_filenames[idx]
            }
        else:
            # Test data - no masks
            image = np.array(image)
            
            if self.transform:
                image = transforms.ToTensor()(image)
                image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
                image = transforms.Grayscale()(image)
            
            sample = {
                'image': image,
                'case_name': self.image_filenames[idx]
            }
        
        return sample

class CustomTransform:
    def __init__(self, img_size):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Resize image and mask
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        
        image = image.resize((self.img_size, self.img_size))
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        if image.shape[0] == 3:  # RGB to grayscale
            image = transforms.Grayscale()(image)
        
        image = transforms.Normalize([0.5], [0.5])(image)
        label = torch.from_numpy(np.array(label)).long()
        
        return {'image': image, 'label': label}

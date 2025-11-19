"""
Image preprocessing utilities for document understanding.
Handles image loading, resizing, normalization, and augmentation.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from .logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Handles all image preprocessing operations."""
    
    def __init__(
        self,
        max_size: int = 1024,
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Initialize image preprocessor.
        
        Args:
            max_size: Maximum dimension (width or height) for resizing
            normalize: Whether to normalize pixel values
            augment: Whether to apply augmentation (for training)
        """
        self.max_size = max_size
        self.normalize = normalize
        self.augment = augment
        
        # Define normalization transform
        self.transform = self._build_transform()
        
        if augment:
            self.augmentation = self._build_augmentation()
    
    def _build_transform(self) -> A.Compose:
        """Build the basic transformation pipeline."""
        transforms = []
        
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def _build_augmentation(self) -> A.Compose:
        """Build augmentation pipeline for training."""
        return A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
        ])
    
    def load_image(
        self,
        image_path: Union[str, Path],
        return_original: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            return_original: Whether to return original image as well
            
        Returns:
            Processed image array, optionally with original
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        original = image_array.copy() if return_original else None
        
        # Preprocess
        processed = self.preprocess(image_array)
        
        if return_original:
            return processed, original
        return processed
    
    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image],
        resize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            resize: Whether to resize the image
            
        Returns:
            Preprocessed image array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Resize if needed
        if resize:
            image = self._resize_image(image)
        
        # Apply augmentation (training only)
        if self.augment:
            image = self.augmentation(image=image)["image"]
        
        # Apply normalization and convert to tensor
        transformed = self.transform(image=image)
        
        return transformed["image"]
    
    def _resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize image maintaining aspect ratio.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = self.max_size / max(h, w)
        
        if scale < 1:
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from ({w}, {h}) to ({new_w}, {new_h})")
        
        return image
    
    def batch_preprocess(
        self,
        images: list,
        to_tensor: bool = True,
    ) -> Union[list, torch.Tensor]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images (arrays or PIL Images)
            to_tensor: Whether to stack into a single tensor
            
        Returns:
            List of preprocessed images or batched tensor
        """
        processed = [self.preprocess(img) for img in images]
        
        if to_tensor and len(processed) > 0:
            return torch.stack(processed)
        
        return processed
    
    @staticmethod
    def denormalize(
        tensor: torch.Tensor,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
    ) -> np.ndarray:
        """
        Denormalize tensor back to displayable image.
        
        Args:
            tensor: Normalized image tensor
            mean: Mean used for normalization
            std: Std used for normalization
            
        Returns:
            Denormalized image array
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        
        # Clamp to [0, 1] and convert to numpy
        tensor = torch.clamp(tensor, 0, 1)
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        
        return (image * 255).astype(np.uint8)

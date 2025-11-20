"""
CRNN-based text recognition.
Lightweight CNN-RNN architecture for text recognition with CTC loss.
"""

import time
from typing import List, Union, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from .base_recognizer import BaseRecognizer, RecognitionResult
from ..detection.base_detector import BoundingBox
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer for sequence modeling."""
    
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        """
        Initialize BiLSTM.
        
        Args:
            n_in: Input features
            n_hidden: Hidden units
            n_out: Output features
        """
        super().__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)
    
    def forward(self, x):
        """Forward pass."""
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        return output


class CRNN(nn.Module):
    """
    CRNN architecture: CNN + RNN + CTC.
    Based on: "An End-to-End Trainable Neural Network for Image-based 
    Sequence Recognition and Its Application to Scene Text Recognition"
    """
    
    def __init__(
        self,
        img_height: int = 32,
        num_channels: int = 1,
        num_classes: int = 37,  # 26 letters + 10 digits + blank
        rnn_hidden: int = 256,
    ):
        """
        Initialize CRNN model.
        
        Args:
            img_height: Height of input images (must be 32)
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes (characters + blank)
            rnn_hidden: Number of hidden units in RNN
        """
        super().__init__()
        
        self.img_height = img_height
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # CNN layers
        self.cnn = nn.Sequential(
            # Conv1: [N, C, 32, W] -> [N, 64, 32, W]
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [N, 64, 16, W/2]
            
            # Conv2: [N, 64, 16, W/2] -> [N, 128, 16, W/2]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [N, 128, 8, W/4]
            
            # Conv3: [N, 128, 8, W/4] -> [N, 256, 8, W/4]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv4: [N, 256, 8, W/4] -> [N, 256, 8, W/4]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # -> [N, 256, 4, W/4]
            
            # Conv5: [N, 256, 4, W/4] -> [N, 512, 4, W/4]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Conv6: [N, 512, 4, W/4] -> [N, 512, 4, W/4]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # -> [N, 512, 2, W/4]
            
            # Conv7: [N, 512, 2, W/4] -> [N, 512, 1, W/4]
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # RNN layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, rnn_hidden, rnn_hidden),
            BidirectionalLSTM(rnn_hidden, rnn_hidden, num_classes),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, C, H, W]
            
        Returns:
            Output tensor [W/4, N, num_classes]
        """
        # CNN feature extraction
        conv = self.cnn(x)  # [N, 512, 1, W/4]
        
        # Reshape for RNN: [N, 512, 1, W/4] -> [N, W/4, 512]
        batch, channel, height, width = conv.size()
        assert height == 1, "Height of conv features must be 1"
        conv = conv.squeeze(2)  # [N, 512, W/4]
        conv = conv.permute(0, 2, 1)  # [N, W/4, 512]
        
        # RNN sequence modeling
        output = self.rnn(conv)  # [N, W/4, num_classes]
        
        # Transpose for CTC: [N, W/4, num_classes] -> [W/4, N, num_classes]
        output = output.permute(1, 0, 2)
        
        return output


class CRNNRecognizer(BaseRecognizer):
    """
    CRNN recognizer for text recognition.
    Lightweight CNN-RNN architecture with CTC loss.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        img_height: int = 32,
        img_width: int = 128,
        num_channels: int = 1,
        alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz",
    ):
        """
        Initialize CRNN recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
            img_height: Height to resize images to
            img_width: Width to resize images to
            num_channels: Number of input channels (1 for grayscale)
            alphabet: Character set for recognition
        """
        super().__init__(model_path=model_path or "crnn_default", device=device)
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.alphabet = alphabet
        
        # Character mapping
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.char_to_idx['<blank>'] = 0  # CTC blank token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.num_classes = len(self.char_to_idx)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
        ])
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load CRNN model."""
        logger.info("Loading CRNN model...")
        
        # Initialize model
        self.model = CRNN(
            img_height=self.img_height,
            num_channels=self.num_channels,
            num_classes=self.num_classes,
        )
        
        # Load checkpoint if provided
        if self.model_path and self.model_path != "crnn_default":
            model_path = Path(self.model_path)
            if model_path.exists():
                logger.info(f"Loading weights from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            else:
                logger.warning(f"Model path {self.model_path} not found. Using random initialization.")
        else:
            logger.warning("No pretrained weights. Using random initialization.")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ CRNN model loaded successfully")
    
    def recognize(
        self,
        image: Union[np.ndarray, Image.Image],
        bboxes: List[BoundingBox],
    ) -> List[RecognitionResult]:
        """
        Recognize text in detected regions.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            
        Returns:
            List of RecognitionResult objects
        """
        if len(bboxes) == 0:
            return []
        
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Crop and recognize each region
        results = []
        for bbox in bboxes:
            # Crop region
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            # Recognize text
            text, confidence = self._recognize_single(cropped)
            
            result = RecognitionResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
            )
            results.append(result)
        
        logger.debug(f"Recognized {len(results)} text regions")
        return results
    
    def _recognize_single(
        self,
        image: Image.Image,
    ) -> Tuple[str, float]:
        """
        Recognize text in a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (text, confidence)
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(img_tensor)  # [W, N, num_classes]
        
        # Decode with greedy algorithm
        text, confidence = self._greedy_decode(output.squeeze(1))  # [W, num_classes]
        
        return text, confidence
    
    def _greedy_decode(
        self,
        output: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Greedy CTC decoder.
        
        Args:
            output: Model output [W, num_classes]
            
        Returns:
            Tuple of (decoded_text, confidence)
        """
        # Get predictions
        probs = torch.softmax(output, dim=1)
        _, preds = torch.max(probs, dim=1)
        preds = preds.cpu().numpy()
        
        # Decode
        decoded = []
        prev_char = None
        confidences = []
        
        for i, pred in enumerate(preds):
            # Skip blanks and repeated characters
            if pred != 0 and pred != prev_char:
                if pred in self.idx_to_char:
                    char = self.idx_to_char[pred]
                    if char != '<blank>':
                        decoded.append(char)
                        confidences.append(probs[i, pred].item())
            prev_char = pred
        
        text = ''.join(decoded)
        confidence = np.mean(confidences) if confidences else 0.0
        
        return text, confidence
    
    def recognize_single(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> str:
        """
        Recognize text in a single image (no bounding box).
        
        Args:
            image: Input image
            
        Returns:
            Recognized text
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        text, _ = self._recognize_single(image)
        return text

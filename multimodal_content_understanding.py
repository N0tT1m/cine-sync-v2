#!/usr/bin/env python3
"""
Multi-Modal Content Understanding for CineSync v2
Implements computer vision for poster analysis and NLP for plot embeddings
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import requests
from PIL import Image
import io
import logging
import pickle
import json
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import (
    BertTokenizer, BertModel,
    CLIPProcessor, CLIPModel,
    pipeline
)
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import colorsys
import webcolors
import re

logger = logging.getLogger(__name__)


class PosterVisualAnalyzer:
    """Computer vision analysis of movie/TV show posters"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        
        # Initialize models
        self.resnet_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # Visual analysis components
        self.color_analyzer = ColorPaletteAnalyzer()
        self.composition_analyzer = CompositionAnalyzer()
        self.style_classifier = VisualStyleClassifier()
        
        # Cache for processed images
        self.image_cache = {}
        self.feature_cache = {}
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_models(self):
        """Lazy loading of heavy models"""
        if self.resnet_model is None:
            logger.info("Loading ResNet50 model for poster analysis...")
            self.resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.resnet_model.eval()
            self.resnet_model = self.resnet_model.to(self.device)
            
            # Remove the final classification layer to get features
            self.resnet_features = nn.Sequential(*list(self.resnet_model.children())[:-1])
        
        if self.clip_model is None:
            logger.info("Loading CLIP model for visual-semantic analysis...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
    
    def analyze_poster(self, poster_url: str, content_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete poster analysis pipeline"""
        
        try:
            # Load and preprocess image
            image = self._load_image_from_url(poster_url)
            if image is None:
                return self._get_default_poster_features()
            
            # Load models if needed
            self._load_models()
            
            analysis_results = {}
            
            # 1. Color palette analysis
            analysis_results['color_features'] = self.color_analyzer.analyze_color_palette(image)
            
            # 2. Visual composition analysis
            analysis_results['composition_features'] = self.composition_analyzer.analyze_composition(image)
            
            # 3. Visual style classification
            analysis_results['style_features'] = self.style_classifier.classify_visual_style(image)
            
            # 4. Deep feature extraction with ResNet
            analysis_results['deep_features'] = self._extract_deep_features(image)
            
            # 5. CLIP-based semantic analysis
            analysis_results['semantic_features'] = self._extract_clip_features(image, content_metadata)
            
            # 6. Genre visual cues detection
            analysis_results['genre_cues'] = self._detect_genre_visual_cues(image, content_metadata)
            
            # 7. Mood and emotional indicators
            analysis_results['mood_features'] = self._analyze_visual_mood(image)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing poster {poster_url}: {e}")
            return self._get_default_poster_features()
    
    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL with caching"""
        
        if url in self.image_cache:
            return self.image_cache[url]
        
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'CineSync-PosterAnalyzer/1.0'
            })
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Cache the image
            self.image_cache[url] = image
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image from {url}: {e}")
            return None
    
    def _extract_deep_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract deep CNN features using ResNet"""
        
        # Preprocess image for ResNet
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet_features(input_tensor)
            features = features.flatten().cpu().numpy()
        
        # Reduce dimensionality and extract meaningful statistics
        feature_stats = {
            'deep_feature_mean': float(np.mean(features)),
            'deep_feature_std': float(np.std(features)),
            'deep_feature_max': float(np.max(features)),
            'deep_feature_min': float(np.min(features)),
            'deep_feature_energy': float(np.sum(features ** 2)),
            'deep_feature_sparsity': float(np.mean(features == 0))
        }
        
        # PCA for dimensionality reduction
        if hasattr(self, 'feature_pca') and self.feature_pca is not None:
            pca_features = self.feature_pca.transform(features.reshape(1, -1))[0]
            for i, val in enumerate(pca_features[:10]):  # Top 10 PCA components
                feature_stats[f'deep_pca_{i}'] = float(val)
        
        return feature_stats
    
    def _extract_clip_features(self, image: Image.Image, content_metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract CLIP-based visual-semantic features"""
        
        clip_features = {}
        
        # Prepare genre-specific text prompts
        genre_prompts = [
            "a dark and mysterious movie poster",
            "a bright and colorful comedy poster", 
            "an action-packed adventure poster",
            "a romantic movie poster",
            "a horror movie poster",
            "a sci-fi futuristic poster",
            "a dramatic movie poster",
            "a family-friendly poster"
        ]
        
        # Process image and text
        inputs = self.clip_processor(
            text=genre_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Extract probabilities for each genre style
        genre_style_names = ['dark_mysterious', 'bright_comedy', 'action_adventure', 
                           'romantic', 'horror', 'sci_fi', 'dramatic', 'family_friendly']
        
        for i, style_name in enumerate(genre_style_names):
            clip_features[f'clip_{style_name}_prob'] = float(probs[0][i])
        
        # Extract raw CLIP image features
        image_features = self.clip_model.get_image_features(**{k: v for k, v in inputs.items() if 'pixel_values' in k})
        image_features = image_features.cpu().numpy()[0]
        
        # Statistical summary of CLIP features
        clip_features.update({
            'clip_feature_mean': float(np.mean(image_features)),
            'clip_feature_std': float(np.std(image_features)),
            'clip_feature_norm': float(np.linalg.norm(image_features))
        })
        
        return clip_features
    
    def _detect_genre_visual_cues(self, image: Image.Image, content_metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Detect visual cues that correlate with specific genres"""
        
        genre_cues = {}
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # 1. Color-based genre indicators
        avg_brightness = np.mean(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
        genre_cues['brightness_level'] = avg_brightness / 255.0
        
        # Dark themes (thriller, horror, crime)
        genre_cues['dark_theme_indicator'] = 1.0 - (avg_brightness / 255.0)
        
        # High contrast (action, thriller)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        genre_cues['high_contrast_indicator'] = min(contrast / 100.0, 1.0)
        
        # 2. Color saturation indicators
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        genre_cues['color_saturation'] = saturation / 255.0
        
        # High saturation (comedy, animation, family)
        genre_cues['vibrant_color_indicator'] = saturation / 255.0
        
        # 3. Red color dominance (action, horror, romance)
        red_dominance = np.mean(img_array[:, :, 0]) / (np.mean(img_array) + 1)
        genre_cues['red_dominance'] = min(red_dominance, 2.0) / 2.0
        
        # 4. Blue color dominance (sci-fi, drama)
        blue_dominance = np.mean(img_array[:, :, 2]) / (np.mean(img_array) + 1)
        genre_cues['blue_dominance'] = min(blue_dominance, 2.0) / 2.0
        
        # 5. Edge density (action vs drama)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        genre_cues['edge_density'] = edge_density
        
        # 6. Symmetry analysis (indicates design sophistication)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        right_half_flipped = np.fliplr(right_half)
        
        if left_half.shape == right_half_flipped.shape:
            symmetry_score = 1 - np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0
            genre_cues['symmetry_score'] = symmetry_score
        else:
            genre_cues['symmetry_score'] = 0.5
        
        return genre_cues
    
    def _analyze_visual_mood(self, image: Image.Image) -> Dict[str, float]:
        """Analyze visual mood indicators from poster"""
        
        mood_features = {}
        img_array = np.array(image)
        
        # 1. Overall brightness (cheerful vs somber)
        brightness = np.mean(img_array) / 255.0
        mood_features['cheerful_indicator'] = brightness
        mood_features['somber_indicator'] = 1.0 - brightness
        
        # 2. Color temperature (warm vs cool)
        red_avg = np.mean(img_array[:, :, 0])
        blue_avg = np.mean(img_array[:, :, 2])
        color_temperature = (red_avg - blue_avg) / 255.0
        mood_features['warm_temperature'] = max(color_temperature, 0)
        mood_features['cool_temperature'] = max(-color_temperature, 0)
        
        # 3. Color harmony (professional vs chaotic)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_std = np.std(hsv[:, :, 0])
        mood_features['color_harmony'] = 1.0 - min(hue_std / 50.0, 1.0)
        
        # 4. Dynamic range (dramatic vs flat)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
        mood_features['dramatic_indicator'] = dynamic_range
        
        # 5. Texture complexity (busy vs minimal)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_complexity = min(laplacian_var / 1000.0, 1.0)
        mood_features['texture_complexity'] = texture_complexity
        mood_features['minimal_design'] = 1.0 - texture_complexity
        
        return mood_features
    
    def _get_default_poster_features(self) -> Dict[str, Any]:
        """Return default features when poster analysis fails"""
        return {
            'color_features': self.color_analyzer._get_default_color_features(),
            'composition_features': self.composition_analyzer._get_default_composition_features(),
            'style_features': self.style_classifier._get_default_style_features(),
            'deep_features': {f'deep_feature_{i}': 0.0 for i in range(10)},
            'semantic_features': {f'clip_feature_{i}': 0.0 for i in range(10)},
            'genre_cues': {f'genre_cue_{i}': 0.0 for i in range(8)},
            'mood_features': {f'mood_feature_{i}': 0.0 for i in range(6)}
        }


class ColorPaletteAnalyzer:
    """Analyze color palettes in movie posters"""
    
    def __init__(self):
        self.color_cache = {}
    
    def analyze_color_palette(self, image: Image.Image) -> Dict[str, float]:
        """Extract comprehensive color palette features"""
        
        img_array = np.array(image)
        color_features = {}
        
        # 1. Dominant colors using K-means
        dominant_colors = self._extract_dominant_colors(img_array, k=5)
        color_features.update(self._analyze_dominant_colors(dominant_colors))
        
        # 2. Color distribution analysis
        color_features.update(self._analyze_color_distribution(img_array))
        
        # 3. Color harmony metrics
        color_features.update(self._analyze_color_harmony(dominant_colors))
        
        # 4. Color temperature and mood
        color_features.update(self._analyze_color_temperature(img_array))
        
        return color_features
    
    def _extract_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        
        # Reshape image to 2D array of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        return [tuple(color) for color in dominant_colors]
    
    def _analyze_dominant_colors(self, colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Analyze properties of dominant colors"""
        
        features = {}
        
        # Convert to different color spaces for analysis
        hsv_colors = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in colors]
        
        # 1. Average properties
        avg_hue = np.mean([hsv[0] for hsv in hsv_colors])
        avg_saturation = np.mean([hsv[1] for hsv in hsv_colors])
        avg_value = np.mean([hsv[2] for hsv in hsv_colors])
        
        features['avg_hue'] = avg_hue
        features['avg_saturation'] = avg_saturation
        features['avg_brightness'] = avg_value
        
        # 2. Color diversity
        hue_std = np.std([hsv[0] for hsv in hsv_colors])
        features['color_diversity'] = hue_std
        
        # 3. Warm vs cool colors
        warm_colors = sum(1 for hsv in hsv_colors if 0.05 < hsv[0] < 0.25 or hsv[0] > 0.8)  # Red/orange/yellow
        cool_colors = sum(1 for hsv in hsv_colors if 0.4 < hsv[0] < 0.75)  # Blue/green
        
        features['warm_color_ratio'] = warm_colors / len(colors)
        features['cool_color_ratio'] = cool_colors / len(colors)
        
        # 4. Color intensity
        high_saturation = sum(1 for hsv in hsv_colors if hsv[1] > 0.6)
        features['high_saturation_ratio'] = high_saturation / len(colors)
        
        return features
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze overall color distribution in the image"""
        
        features = {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 1. Color channel statistics
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            channel_data = hsv[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data) / 255.0
            features[f'{channel}_std'] = np.std(channel_data) / 255.0
            features[f'{channel}_range'] = (np.max(channel_data) - np.min(channel_data)) / 255.0
        
        # 2. RGB channel analysis
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = img_array[:, :, i].flatten()
            features[f'{channel}_dominance'] = np.mean(channel_data) / 255.0
        
        return features
    
    def _analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Analyze color harmony relationships"""
        
        features = {}
        
        if len(colors) < 2:
            return {'color_harmony_score': 0.5}
        
        # Convert to HSV for harmony analysis
        hsv_colors = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in colors]
        
        # 1. Complementary colors (opposite on color wheel)
        complementary_pairs = 0
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                hue_diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                if 0.45 < hue_diff < 0.55:  # Approximately opposite
                    complementary_pairs += 1
        
        features['complementary_harmony'] = complementary_pairs / (len(colors) * (len(colors) - 1) / 2)
        
        # 2. Analogous colors (adjacent on color wheel)
        analogous_pairs = 0
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                hue_diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                if hue_diff < 0.1 or hue_diff > 0.9:  # Close on circular hue wheel
                    analogous_pairs += 1
        
        features['analogous_harmony'] = analogous_pairs / (len(colors) * (len(colors) - 1) / 2)
        
        # 3. Overall harmony score
        hues = [hsv[0] for hsv in hsv_colors]
        hue_variance = np.var(hues)
        features['color_harmony_score'] = 1.0 / (1.0 + hue_variance * 10)  # Higher variance = lower harmony
        
        return features
    
    def _analyze_color_temperature(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze color temperature and mood indicators"""
        
        features = {}
        
        # Calculate color temperature based on red/blue ratio
        red_avg = np.mean(img_array[:, :, 0])
        blue_avg = np.mean(img_array[:, :, 2])
        
        if blue_avg > 0:
            color_temperature = red_avg / blue_avg
            features['color_temperature'] = min(color_temperature / 2.0, 1.0)  # Normalize
        else:
            features['color_temperature'] = 0.5
        
        # Warm vs cool classification
        features['warm_temperature'] = 1.0 if red_avg > blue_avg else 0.0
        features['cool_temperature'] = 1.0 if blue_avg > red_avg else 0.0
        
        return features
    
    def _get_default_color_features(self) -> Dict[str, float]:
        """Default color features when analysis fails"""
        return {
            'avg_hue': 0.5,
            'avg_saturation': 0.5,
            'avg_brightness': 0.5,
            'color_diversity': 0.5,
            'warm_color_ratio': 0.4,
            'cool_color_ratio': 0.4,
            'high_saturation_ratio': 0.3,
            'hue_mean': 0.5,
            'saturation_mean': 0.5,
            'value_mean': 0.5,
            'red_dominance': 0.33,
            'green_dominance': 0.33,
            'blue_dominance': 0.33,
            'complementary_harmony': 0.2,
            'analogous_harmony': 0.3,
            'color_harmony_score': 0.5,
            'color_temperature': 0.5,
            'warm_temperature': 0.5,
            'cool_temperature': 0.5
        }


class CompositionAnalyzer:
    """Analyze visual composition of posters"""
    
    def analyze_composition(self, image: Image.Image) -> Dict[str, float]:
        """Analyze poster composition and layout"""
        
        img_array = np.array(image)
        composition_features = {}
        
        # 1. Rule of thirds analysis
        composition_features.update(self._analyze_rule_of_thirds(img_array))
        
        # 2. Symmetry analysis
        composition_features.update(self._analyze_symmetry(img_array))
        
        # 3. Visual weight distribution
        composition_features.update(self._analyze_visual_weight(img_array))
        
        # 4. Text vs image area ratio
        composition_features.update(self._analyze_text_image_ratio(img_array))
        
        return composition_features
    
    def _analyze_rule_of_thirds(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze adherence to rule of thirds"""
        
        height, width = img_array.shape[:2]
        
        # Define thirds grid
        h_thirds = [height // 3, 2 * height // 3]
        w_thirds = [width // 3, 2 * width // 3]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density at third lines
        h_line_density = (np.sum(edges[h_thirds[0]-5:h_thirds[0]+5, :]) + 
                         np.sum(edges[h_thirds[1]-5:h_thirds[1]+5, :])) / (width * 10)
        
        w_line_density = (np.sum(edges[:, w_thirds[0]-5:w_thirds[0]+5]) + 
                         np.sum(edges[:, w_thirds[1]-5:w_thirds[1]+5])) / (height * 10)
        
        total_edge_density = np.sum(edges) / (height * width)
        
        return {
            'rule_of_thirds_adherence': (h_line_density + w_line_density) / (total_edge_density + 1e-6),
            'horizontal_third_alignment': h_line_density / (total_edge_density + 1e-6),
            'vertical_third_alignment': w_line_density / (total_edge_density + 1e-6)
        }
    
    def _analyze_symmetry(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze symmetry in composition"""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # Vertical symmetry
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        right_flipped = np.fliplr(right_half)
        
        if left_half.shape == right_flipped.shape:
            vertical_symmetry = 1 - np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float))) / 255
        else:
            vertical_symmetry = 0.5
        
        # Horizontal symmetry
        top_half = gray[:height//2, :]
        bottom_half = gray[height//2:, :]
        bottom_flipped = np.flipud(bottom_half)
        
        if top_half.shape == bottom_flipped.shape:
            horizontal_symmetry = 1 - np.mean(np.abs(top_half.astype(float) - bottom_flipped.astype(float))) / 255
        else:
            horizontal_symmetry = 0.5
        
        return {
            'vertical_symmetry': vertical_symmetry,
            'horizontal_symmetry': horizontal_symmetry,
            'overall_symmetry': (vertical_symmetry + horizontal_symmetry) / 2
        }
    
    def _analyze_visual_weight(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of visual weight"""
        
        # Convert to grayscale and calculate visual weight (brightness + contrast)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate local contrast
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contrast = np.abs(gray.astype(float) - blur.astype(float))
        
        # Visual weight combines brightness and contrast
        visual_weight = (gray.astype(float) / 255) * 0.7 + (contrast / 255) * 0.3
        
        height, width = visual_weight.shape
        
        # Divide into quadrants
        top_left = visual_weight[:height//2, :width//2]
        top_right = visual_weight[:height//2, width//2:]
        bottom_left = visual_weight[height//2:, :width//2]
        bottom_right = visual_weight[height//2:, width//2:]
        
        quadrant_weights = [
            np.mean(top_left),
            np.mean(top_right),
            np.mean(bottom_left),
            np.mean(bottom_right)
        ]
        
        # Calculate balance
        left_weight = (quadrant_weights[0] + quadrant_weights[2]) / 2
        right_weight = (quadrant_weights[1] + quadrant_weights[3]) / 2
        top_weight = (quadrant_weights[0] + quadrant_weights[1]) / 2
        bottom_weight = (quadrant_weights[2] + quadrant_weights[3]) / 2
        
        return {
            'left_visual_weight': left_weight,
            'right_visual_weight': right_weight,
            'top_visual_weight': top_weight,
            'bottom_visual_weight': bottom_weight,
            'horizontal_balance': 1 - abs(left_weight - right_weight),
            'vertical_balance': 1 - abs(top_weight - bottom_weight),
            'visual_weight_variance': np.var(quadrant_weights)
        }
    
    def _analyze_text_image_ratio(self, img_array: np.ndarray) -> Dict[str, float]:
        """Estimate text vs image content ratio"""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Simple text detection using edge density and horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Horizontal lines often indicate text
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Estimate text area
        text_pixels = np.sum(horizontal_lines > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        text_ratio = text_pixels / total_pixels
        
        return {
            'estimated_text_ratio': text_ratio,
            'estimated_image_ratio': 1 - text_ratio,
            'text_heavy': 1.0 if text_ratio > 0.3 else 0.0,
            'image_focused': 1.0 if text_ratio < 0.1 else 0.0
        }
    
    def _get_default_composition_features(self) -> Dict[str, float]:
        """Default composition features"""
        return {
            'rule_of_thirds_adherence': 0.5,
            'horizontal_third_alignment': 0.3,
            'vertical_third_alignment': 0.3,
            'vertical_symmetry': 0.5,
            'horizontal_symmetry': 0.5,
            'overall_symmetry': 0.5,
            'left_visual_weight': 0.5,
            'right_visual_weight': 0.5,
            'top_visual_weight': 0.5,
            'bottom_visual_weight': 0.5,
            'horizontal_balance': 0.7,
            'vertical_balance': 0.7,
            'visual_weight_variance': 0.1,
            'estimated_text_ratio': 0.2,
            'estimated_image_ratio': 0.8,
            'text_heavy': 0.0,
            'image_focused': 1.0
        }


class VisualStyleClassifier:
    """Classify visual style of posters"""
    
    def classify_visual_style(self, image: Image.Image) -> Dict[str, float]:
        """Classify the visual style of the poster"""
        
        img_array = np.array(image)
        style_features = {}
        
        # 1. Minimalist vs Complex
        style_features.update(self._classify_complexity(img_array))
        
        # 2. Modern vs Vintage
        style_features.update(self._classify_era_style(img_array))
        
        # 3. Photographic vs Illustrated
        style_features.update(self._classify_medium_style(img_array))
        
        # 4. Professional vs Amateur
        style_features.update(self._classify_quality_level(img_array))
        
        return style_features
    
    def _classify_complexity(self, img_array: np.ndarray) -> Dict[str, float]:
        """Classify minimalist vs complex design"""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge density as complexity indicator
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Color complexity
        colors_used = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_complexity = min(colors_used / 1000, 1.0)  # Normalize
        
        # Texture complexity
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_complexity = min(laplacian_var / 1000, 1.0)
        
        overall_complexity = (edge_density + color_complexity + texture_complexity) / 3
        
        return {
            'minimalist_style': 1 - overall_complexity,
            'complex_style': overall_complexity,
            'edge_complexity': edge_density,
            'color_complexity': color_complexity,
            'texture_complexity': texture_complexity
        }
    
    def _classify_era_style(self, img_array: np.ndarray) -> Dict[str, float]:
        """Classify modern vs vintage style"""
        
        # Color saturation and brightness indicators
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1]) / 255
        avg_brightness = np.mean(hsv[:, :, 2]) / 255
        
        # Modern typically has high saturation and brightness
        modern_indicator = (avg_saturation + avg_brightness) / 2
        
        # Vintage indicators: sepia tones, lower saturation
        red_avg = np.mean(img_array[:, :, 0])
        green_avg = np.mean(img_array[:, :, 1])
        blue_avg = np.mean(img_array[:, :, 2])
        
        # Sepia tone detection (yellow/brown dominance)
        sepia_indicator = 0.0
        if green_avg > 0 and blue_avg > 0:
            sepia_indicator = (red_avg / green_avg + red_avg / blue_avg) / 2 - 1
            sepia_indicator = max(0, min(sepia_indicator, 1))
        
        vintage_indicator = (sepia_indicator + (1 - avg_saturation)) / 2
        
        return {
            'modern_style': modern_indicator,
            'vintage_style': vintage_indicator,
            'sepia_tone': sepia_indicator,
            'high_saturation': avg_saturation
        }
    
    def _classify_medium_style(self, img_array: np.ndarray) -> Dict[str, float]:
        """Classify photographic vs illustrated style"""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Noise characteristics (photos have more noise)
        noise_level = np.std(gray)
        photo_noise_indicator = min(noise_level / 50, 1.0)
        
        # Gradient smoothness (illustrations often have smoother gradients)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_smoothness = 1 - (np.std(gradient_magnitude) / np.mean(gradient_magnitude + 1))
        
        # Color quantization (illustrations often have fewer distinct colors)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_quantization = 1 - min(unique_colors / 10000, 1.0)
        
        illustration_indicator = (gradient_smoothness + color_quantization) / 2
        photographic_indicator = (photo_noise_indicator + (1 - color_quantization)) / 2
        
        return {
            'photographic_style': photographic_indicator,
            'illustrated_style': illustration_indicator,
            'noise_level': photo_noise_indicator,
            'gradient_smoothness': gradient_smoothness,
            'color_quantization': color_quantization
        }
    
    def _classify_quality_level(self, img_array: np.ndarray) -> Dict[str, float]:
        """Classify professional vs amateur quality"""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (professional images tend to be sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 500, 1.0)
        
        # Color balance (professional images have better color balance)
        r_avg = np.mean(img_array[:, :, 0])
        g_avg = np.mean(img_array[:, :, 1])
        b_avg = np.mean(img_array[:, :, 2])
        
        color_balance = 1 - (np.std([r_avg, g_avg, b_avg]) / np.mean([r_avg, g_avg, b_avg]))
        
        # Composition quality (rule of thirds, symmetry)
        height, width = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for centered composition (amateur) vs off-center (professional)
        center_region = edges[height//3:2*height//3, width//3:2*width//3]
        center_density = np.sum(center_region > 0) / (center_region.shape[0] * center_region.shape[1])
        off_center_composition = 1 - center_density
        
        professional_indicator = (sharpness + color_balance + off_center_composition) / 3
        
        return {
            'professional_quality': professional_indicator,
            'amateur_quality': 1 - professional_indicator,
            'sharpness_level': sharpness,
            'color_balance': color_balance,
            'composition_sophistication': off_center_composition
        }
    
    def _get_default_style_features(self) -> Dict[str, float]:
        """Default style features"""
        return {
            'minimalist_style': 0.5,
            'complex_style': 0.5,
            'modern_style': 0.7,
            'vintage_style': 0.3,
            'photographic_style': 0.6,
            'illustrated_style': 0.4,
            'professional_quality': 0.7,
            'amateur_quality': 0.3,
            'sharpness_level': 0.6,
            'color_balance': 0.7,
            'composition_sophistication': 0.6
        }


class PlotSummaryAnalyzer:
    """NLP analysis of plot summaries using BERT and other transformers"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        
        # Initialize models (lazy loading)
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentiment_analyzer = None
        self.theme_classifier = None
        
        # Analysis components
        self.theme_extractor = ThemeExtractor()
        self.character_analyzer = CharacterAnalyzer()
        self.narrative_analyzer = NarrativeStructureAnalyzer()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_models(self):
        """Lazy loading of NLP models"""
        if self.bert_model is None:
            logger.info("Loading BERT model for plot analysis...")
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model = self.bert_model.to(self.device)
            self.bert_model.eval()
        
        if self.sentiment_analyzer is None:
            logger.info("Loading sentiment analysis pipeline...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device.type == 'cuda' else -1
            )
    
    def analyze_plot_summary(self, plot_text: str, content_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete plot summary analysis"""
        
        if not plot_text or len(plot_text.strip()) == 0:
            return self._get_default_plot_features()
        
        try:
            self._load_models()
            
            analysis_results = {}
            
            # 1. BERT embeddings for semantic understanding
            analysis_results['bert_features'] = self._extract_bert_embeddings(plot_text)
            
            # 2. Theme extraction and analysis
            analysis_results['theme_features'] = self.theme_extractor.extract_themes(plot_text)
            
            # 3. Character archetype identification
            analysis_results['character_features'] = self.character_analyzer.analyze_characters(plot_text)
            
            # 4. Narrative structure analysis
            analysis_results['narrative_features'] = self.narrative_analyzer.analyze_structure(plot_text)
            
            # 5. Emotional arc analysis
            analysis_results['emotion_features'] = self._analyze_emotional_arc(plot_text)
            
            # 6. Genre prediction from plot
            analysis_results['genre_prediction'] = self._predict_genre_from_plot(plot_text, content_metadata)
            
            # 7. Content complexity analysis
            analysis_results['complexity_features'] = self._analyze_content_complexity(plot_text)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing plot summary: {e}")
            return self._get_default_plot_features()
    
    def _extract_bert_embeddings(self, text: str) -> Dict[str, float]:
        """Extract BERT embeddings from plot text"""
        
        # Tokenize and encode
        inputs = self.bert_tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
            # Use [CLS] token embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            # Also get mean pooling of all tokens
            token_embeddings = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()  # Exclude [CLS] and [SEP]
            mean_embedding = np.mean(token_embeddings, axis=0)
        
        # Extract statistical features from embeddings
        bert_features = {
            'bert_cls_mean': float(np.mean(cls_embedding)),
            'bert_cls_std': float(np.std(cls_embedding)),
            'bert_cls_max': float(np.max(cls_embedding)),
            'bert_cls_min': float(np.min(cls_embedding)),
            'bert_mean_pool_mean': float(np.mean(mean_embedding)),
            'bert_mean_pool_std': float(np.std(mean_embedding)),
            'bert_embedding_norm': float(np.linalg.norm(cls_embedding)),
            'bert_embedding_sparsity': float(np.mean(cls_embedding == 0))
        }
        
        # PCA for dimensionality reduction (top 10 components)
        if hasattr(self, 'bert_pca') and self.bert_pca is not None:
            pca_features = self.bert_pca.transform(cls_embedding.reshape(1, -1))[0]
            for i, val in enumerate(pca_features[:10]):
                bert_features[f'bert_pca_{i}'] = float(val)
        
        return bert_features
    
    def _analyze_emotional_arc(self, text: str) -> Dict[str, float]:
        """Analyze the emotional progression in the plot"""
        
        emotion_features = {}
        
        # Split text into segments for arc analysis
        sentences = text.split('. ')
        if len(sentences) < 3:
            # Text too short for arc analysis
            emotion_features['emotional_arc_intensity'] = 0.5
            emotion_features['emotional_stability'] = 0.7
            emotion_features['negative_emotion_ratio'] = 0.3
            emotion_features['positive_emotion_ratio'] = 0.5
            return emotion_features
        
        # Analyze sentiment of each segment
        segment_sentiments = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short segments
                try:
                    sentiment_result = self.sentiment_analyzer(sentence)[0]
                    
                    # Convert to numeric score
                    if sentiment_result['label'] == 'LABEL_2':  # Positive
                        score = sentiment_result['score']
                    elif sentiment_result['label'] == 'LABEL_0':  # Negative
                        score = -sentiment_result['score']
                    else:  # Neutral
                        score = 0.0
                    
                    segment_sentiments.append(score)
                except:
                    segment_sentiments.append(0.0)
        
        if not segment_sentiments:
            return emotion_features
        
        # Calculate emotional arc features
        emotion_features['emotional_arc_intensity'] = np.std(segment_sentiments)
        emotion_features['emotional_stability'] = 1.0 - np.std(segment_sentiments)
        
        # Overall emotional tone
        avg_sentiment = np.mean(segment_sentiments)
        emotion_features['overall_emotional_tone'] = (avg_sentiment + 1) / 2  # Normalize to 0-1
        
        # Positive/negative emotion ratios
        positive_segments = sum(1 for s in segment_sentiments if s > 0)
        negative_segments = sum(1 for s in segment_sentiments if s < 0)
        total_segments = len(segment_sentiments)
        
        emotion_features['positive_emotion_ratio'] = positive_segments / total_segments
        emotion_features['negative_emotion_ratio'] = negative_segments / total_segments
        
        # Emotional trajectory (beginning to end)
        if len(segment_sentiments) >= 3:
            beginning_sentiment = np.mean(segment_sentiments[:len(segment_sentiments)//3])
            ending_sentiment = np.mean(segment_sentiments[-len(segment_sentiments)//3:])
            emotion_features['emotional_trajectory'] = (ending_sentiment - beginning_sentiment + 2) / 4  # Normalize
        else:
            emotion_features['emotional_trajectory'] = 0.5
        
        return emotion_features
    
    def _predict_genre_from_plot(self, text: str, content_metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Predict genre based on plot content"""
        
        # Genre keywords and patterns
        genre_indicators = {
            'action': ['fight', 'battle', 'chase', 'explosion', 'combat', 'weapon', 'war', 'soldier'],
            'comedy': ['funny', 'laugh', 'humor', 'joke', 'comic', 'hilarious', 'amusing', 'witty'],
            'drama': ['emotional', 'relationship', 'family', 'life', 'struggle', 'personal', 'character'],
            'horror': ['scary', 'monster', 'dead', 'ghost', 'evil', 'terror', 'nightmare', 'haunted'],
            'romance': ['love', 'romantic', 'relationship', 'marry', 'couple', 'heart', 'kiss', 'passion'],
            'thriller': ['suspense', 'mystery', 'danger', 'secret', 'conspiracy', 'tension', 'pursue'],
            'sci_fi': ['future', 'space', 'alien', 'technology', 'robot', 'scientific', 'universe'],
            'fantasy': ['magic', 'wizard', 'dragon', 'mythical', 'fantasy', 'supernatural', 'enchanted'],
            'crime': ['detective', 'murder', 'criminal', 'police', 'investigation', 'crime', 'law'],
            'documentary': ['real', 'documentary', 'true story', 'factual', 'historical', 'biography']
        }
        
        text_lower = text.lower()
        genre_scores = {}
        
        for genre, keywords in genre_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length and keyword count
            genre_scores[f'plot_genre_{genre}'] = score / (len(keywords) * max(len(text.split()) / 100, 1))
        
        # Normalize scores
        total_score = sum(genre_scores.values())
        if total_score > 0:
            for key in genre_scores:
                genre_scores[key] = genre_scores[key] / total_score
        
        return genre_scores
    
    def _analyze_content_complexity(self, text: str) -> Dict[str, float]:
        """Analyze complexity of plot content"""
        
        complexity_features = {}
        
        # Text statistics
        words = text.split()
        sentences = text.split('.')
        
        # Basic readability metrics
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = np.mean([len(word) for word in words])
        
        complexity_features['avg_sentence_length'] = min(avg_words_per_sentence / 20, 1.0)  # Normalize
        complexity_features['avg_word_length'] = min(avg_chars_per_word / 10, 1.0)  # Normalize
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / max(len(words), 1)
        complexity_features['vocabulary_diversity'] = vocabulary_diversity
        
        # Complex sentence structures (approximated by punctuation)
        complex_punctuation = text.count(',') + text.count(';') + text.count(':')
        complexity_features['sentence_complexity'] = min(complex_punctuation / max(len(sentences), 1) / 3, 1.0)
        
        # Overall complexity score
        complexity_features['plot_complexity'] = (
            complexity_features['avg_sentence_length'] * 0.3 +
            complexity_features['avg_word_length'] * 0.2 +
            complexity_features['vocabulary_diversity'] * 0.3 +
            complexity_features['sentence_complexity'] * 0.2
        )
        
        return complexity_features
    
    def _get_default_plot_features(self) -> Dict[str, Any]:
        """Default plot features when analysis fails"""
        return {
            'bert_features': {f'bert_feature_{i}': 0.0 for i in range(15)},
            'theme_features': {f'theme_{i}': 0.0 for i in range(10)},
            'character_features': {f'character_{i}': 0.0 for i in range(8)},
            'narrative_features': {f'narrative_{i}': 0.0 for i in range(6)},
            'emotion_features': {
                'emotional_arc_intensity': 0.5,
                'emotional_stability': 0.7,
                'overall_emotional_tone': 0.5,
                'positive_emotion_ratio': 0.5,
                'negative_emotion_ratio': 0.3,
                'emotional_trajectory': 0.5
            },
            'genre_prediction': {f'plot_genre_{genre}': 0.1 for genre in 
                               ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 'sci_fi', 'fantasy', 'crime', 'documentary']},
            'complexity_features': {
                'avg_sentence_length': 0.5,
                'avg_word_length': 0.5,
                'vocabulary_diversity': 0.6,
                'sentence_complexity': 0.4,
                'plot_complexity': 0.5
            }
        }


class ThemeExtractor:
    """Extract themes from plot summaries"""
    
    def extract_themes(self, text: str) -> Dict[str, float]:
        """Extract thematic elements from plot text"""
        
        # Common themes and their indicators
        theme_patterns = {
            'redemption': ['redeem', 'forgiveness', 'second chance', 'atone', 'salvation'],
            'love': ['love', 'romance', 'relationship', 'heart', 'affection'],
            'betrayal': ['betray', 'deceive', 'backstab', 'treachery', 'double-cross'],
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution', 'avenge'],
            'coming_of_age': ['grow up', 'mature', 'adolescent', 'teenager', 'youth'],
            'sacrifice': ['sacrifice', 'give up', 'selfless', 'noble', 'heroic'],
            'power': ['power', 'control', 'dominate', 'authority', 'rule'],
            'survival': ['survive', 'endure', 'overcome', 'struggle', 'persevere'],
            'identity': ['identity', 'who am i', 'self-discovery', 'find yourself'],
            'family': ['family', 'parent', 'child', 'sibling', 'relative']
        }
        
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, patterns in theme_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            theme_scores[f'theme_{theme}'] = score / len(patterns)
        
        return theme_scores


class CharacterAnalyzer:
    """Analyze character archetypes in plot summaries"""
    
    def analyze_characters(self, text: str) -> Dict[str, float]:
        """Identify character archetypes from plot description"""
        
        archetype_indicators = {
            'hero': ['hero', 'protagonist', 'champion', 'savior', 'main character'],
            'villain': ['villain', 'antagonist', 'evil', 'bad guy', 'enemy'],
            'mentor': ['mentor', 'teacher', 'guide', 'wise', 'elder'],
            'love_interest': ['love interest', 'romantic', 'girlfriend', 'boyfriend', 'partner'],
            'comic_relief': ['funny', 'comic', 'humorous', 'sidekick', 'comedy'],
            'innocent': ['innocent', 'naive', 'pure', 'child-like', 'vulnerable'],
            'rebel': ['rebel', 'outlaw', 'rogue', 'maverick', 'troublemaker'],
            'everyman': ['ordinary', 'normal', 'average', 'common', 'regular']
        }
        
        text_lower = text.lower()
        character_features = {}
        
        for archetype, indicators in archetype_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            character_features[f'character_{archetype}'] = min(score / 2, 1.0)  # Normalize
        
        return character_features


class NarrativeStructureAnalyzer:
    """Analyze narrative structure elements"""
    
    def analyze_structure(self, text: str) -> Dict[str, float]:
        """Analyze narrative structure indicators"""
        
        structure_features = {}
        
        # Narrative indicators
        flashback_indicators = ['remember', 'flashback', 'years ago', 'in the past', 'memory']
        twist_indicators = ['twist', 'reveal', 'discover', 'unexpected', 'surprise', 'plot twist']
        mystery_indicators = ['mystery', 'unknown', 'secret', 'hidden', 'investigate']
        journey_indicators = ['journey', 'travel', 'adventure', 'quest', 'expedition']
        conflict_indicators = ['conflict', 'fight', 'battle', 'struggle', 'oppose']
        resolution_indicators = ['resolve', 'end', 'conclusion', 'finally', 'ultimately']
        
        text_lower = text.lower()
        
        structure_features['narrative_flashback'] = sum(1 for i in flashback_indicators if i in text_lower) / len(flashback_indicators)
        structure_features['narrative_twist'] = sum(1 for i in twist_indicators if i in text_lower) / len(twist_indicators)
        structure_features['narrative_mystery'] = sum(1 for i in mystery_indicators if i in text_lower) / len(mystery_indicators)
        structure_features['narrative_journey'] = sum(1 for i in journey_indicators if i in text_lower) / len(journey_indicators)
        structure_features['narrative_conflict'] = sum(1 for i in conflict_indicators if i in text_lower) / len(conflict_indicators)
        structure_features['narrative_resolution'] = sum(1 for i in resolution_indicators if i in text_lower) / len(resolution_indicators)
        
        return structure_features


class MultiModalContentAnalyzer:
    """Complete multi-modal content understanding system"""
    
    def __init__(self, device: str = 'auto'):
        self.poster_analyzer = PosterVisualAnalyzer(device)
        self.plot_analyzer = PlotSummaryAnalyzer(device)
        
        # Feature fusion components
        self.feature_fusion = ContentFeatureFusion()
        
    def analyze_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete multi-modal content analysis"""
        
        analysis_results = {
            'content_id': content_data.get('content_id'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Poster analysis
        poster_url = content_data.get('poster_url')
        if poster_url:
            logger.info(f"Analyzing poster: {poster_url}")
            poster_features = self.poster_analyzer.analyze_poster(poster_url, content_data)
            analysis_results['poster_analysis'] = poster_features
        else:
            analysis_results['poster_analysis'] = self.poster_analyzer._get_default_poster_features()
        
        # 2. Plot analysis
        plot_summary = content_data.get('plot_summary', content_data.get('overview', ''))
        if plot_summary:
            logger.info("Analyzing plot summary...")
            plot_features = self.plot_analyzer.analyze_plot_summary(plot_summary, content_data)
            analysis_results['plot_analysis'] = plot_features
        else:
            analysis_results['plot_analysis'] = self.plot_analyzer._get_default_plot_features()
        
        # 3. Feature fusion
        fused_features = self.feature_fusion.fuse_features(
            analysis_results['poster_analysis'],
            analysis_results['plot_analysis'],
            content_data
        )
        analysis_results['fused_features'] = fused_features
        
        return analysis_results
    
    def extract_feature_vector(self, content_data: Dict[str, Any]) -> np.ndarray:
        """Extract unified feature vector for ML models"""
        
        analysis = self.analyze_content(content_data)
        return self.feature_fusion.create_feature_vector(analysis)
    
    def save_analysis(self, analysis_results: Dict[str, Any], filepath: str):
        """Save analysis results to file"""
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]:
        """Load analysis results from file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class ContentFeatureFusion:
    """Fuse visual and textual features for unified representation"""
    
    def fuse_features(self, poster_features: Dict[str, Any], plot_features: Dict[str, Any], 
                     content_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Fuse multi-modal features into unified representation"""
        
        fused_features = {}
        
        # 1. Direct feature inclusion
        fused_features.update(self._flatten_features(poster_features, 'visual'))
        fused_features.update(self._flatten_features(plot_features, 'textual'))
        
        # 2. Cross-modal feature interactions
        fused_features.update(self._compute_cross_modal_features(poster_features, plot_features))
        
        # 3. Metadata integration
        fused_features.update(self._integrate_metadata(content_metadata))
        
        return fused_features
    
    def _flatten_features(self, features: Dict[str, Any], prefix: str) -> Dict[str, float]:
        """Flatten nested feature dictionaries"""
        flattened = {}
        
        def _flatten_recursive(obj, parent_key=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}_{key}" if parent_key else key
                    _flatten_recursive(value, new_key)
            else:
                # Convert to float if possible
                try:
                    flattened[f"{prefix}_{parent_key}"] = float(obj)
                except (ValueError, TypeError):
                    flattened[f"{prefix}_{parent_key}"] = 0.0
        
        _flatten_recursive(features)
        return flattened
    
    def _compute_cross_modal_features(self, poster_features: Dict[str, Any], 
                                    plot_features: Dict[str, Any]) -> Dict[str, float]:
        """Compute features that combine visual and textual information"""
        
        cross_modal = {}
        
        # Visual-genre alignment
        poster_mood = poster_features.get('mood_features', {})
        plot_genres = plot_features.get('genre_prediction', {})
        
        # Dark visual + horror/thriller genre alignment
        dark_visual = poster_mood.get('somber_indicator', 0.5)
        dark_genres = plot_genres.get('plot_genre_horror', 0) + plot_genres.get('plot_genre_thriller', 0)
        cross_modal['dark_visual_genre_alignment'] = dark_visual * dark_genres
        
        # Bright visual + comedy genre alignment
        bright_visual = poster_mood.get('cheerful_indicator', 0.5)
        comedy_genre = plot_genres.get('plot_genre_comedy', 0)
        cross_modal['bright_comedy_alignment'] = bright_visual * comedy_genre
        
        # Action visual cues + action plot
        visual_action = poster_features.get('genre_cues', {}).get('high_contrast_indicator', 0.5)
        plot_action = plot_genres.get('plot_genre_action', 0)
        cross_modal['action_visual_plot_alignment'] = visual_action * plot_action
        
        # Romance color warmth + romance plot
        color_warmth = poster_features.get('color_features', {}).get('warm_temperature', 0.5)
        romance_plot = plot_genres.get('plot_genre_romance', 0)
        cross_modal['romance_warmth_alignment'] = color_warmth * romance_plot
        
        # Visual complexity + plot complexity
        visual_complexity = poster_features.get('style_features', {}).get('complex_style', 0.5)
        plot_complexity = plot_features.get('complexity_features', {}).get('plot_complexity', 0.5)
        cross_modal['overall_complexity_alignment'] = visual_complexity * plot_complexity
        
        return cross_modal
    
    def _integrate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Integrate content metadata as features"""
        
        meta_features = {}
        
        # Release year (normalized)
        year = metadata.get('release_year', metadata.get('year', 2020))
        if year:
            meta_features['content_age'] = (2024 - int(year)) / 100  # Normalize
            meta_features['is_recent'] = 1.0 if (2024 - int(year)) < 5 else 0.0
        
        # Content type
        content_type = metadata.get('content_type', metadata.get('type', 'unknown'))
        meta_features['is_movie'] = 1.0 if content_type == 'movie' else 0.0
        meta_features['is_tv_show'] = 1.0 if content_type in ['tv', 'series'] else 0.0
        
        # Runtime (if available)
        runtime = metadata.get('runtime', metadata.get('duration', 0))
        if runtime:
            meta_features['runtime_normalized'] = min(int(runtime) / 300, 1.0)  # Normalize by 5 hours
            meta_features['is_long_content'] = 1.0 if int(runtime) > 150 else 0.0
        
        # Rating information
        rating = metadata.get('rating', metadata.get('vote_average', 0))
        if rating:
            meta_features['content_rating'] = float(rating) / 10.0  # Normalize to 0-1
        
        return meta_features
    
    def create_feature_vector(self, analysis_results: Dict[str, Any]) -> np.ndarray:
        """Create unified feature vector for ML models"""
        
        fused_features = analysis_results.get('fused_features', {})
        
        # Sort features for consistent ordering
        feature_names = sorted(fused_features.keys())
        feature_vector = [fused_features[name] for name in feature_names]
        
        return np.array(feature_vector, dtype=np.float32)


# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-modal analyzer
    analyzer = MultiModalContentAnalyzer()
    
    # Example content data
    example_content = {
        'content_id': 12345,
        'title': 'The Dark Knight',
        'poster_url': 'https://example.com/dark_knight_poster.jpg',
        'plot_summary': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
        'release_year': 2008,
        'content_type': 'movie',
        'runtime': 152,
        'genres': ['Action', 'Crime', 'Drama'],
        'rating': 9.0
    }
    
    # Analyze content (note: poster URL is fake, will use defaults)
    print("Analyzing multi-modal content...")
    analysis_results = analyzer.analyze_content(example_content)
    
    print("\nAnalysis completed!")
    print(f"Poster analysis features: {len(analysis_results['poster_analysis'])}")
    print(f"Plot analysis features: {len(analysis_results['plot_analysis'])}")
    print(f"Fused features: {len(analysis_results['fused_features'])}")
    
    # Extract feature vector
    feature_vector = analyzer.extract_feature_vector(example_content)
    print(f"\nUnified feature vector shape: {feature_vector.shape}")
    print(f"Feature vector sample: {feature_vector[:10]}")
    
    # Show some specific features
    fused_features = analysis_results['fused_features']
    print(f"\nSample fused features:")
    for i, (key, value) in enumerate(sorted(fused_features.items())[:15]):
        print(f"  {key}: {value:.4f}")
    
    print("\nMulti-modal content understanding analysis complete!")
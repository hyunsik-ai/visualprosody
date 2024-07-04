#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:09:28 2024

@author: khs
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 23:31:47 2024

@author: khs
"""

import timm
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import matplotlib
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import wave
# Configure logging
#logging.basicConfig(filename='training_new_NN_Swin.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model name
model_name = "CA"  # or any other model name
deepface = "deepface"

# Configure logging with the model name
# Configure logging with the model name
logging.basicConfig(
    filename=f'training_new_NN_{model_name}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - ' + model_name,
)

class VideoPitchDataset(Dataset):
    def __init__(self, base_dir, target_type='pitch', num_frames=None):
        self.base_dir = base_dir
        self.rawfile_dir = base_dir + "/rawfile"
        self.target_type = target_type
        self.video_files = []
        self.feature_files = {}
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.num_frames = num_frames
        self._load_data()

    def _load_data(self):
        for subdir in os.listdir(self.rawfile_dir):
            subdir_path = os.path.join(self.rawfile_dir, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith(".mkv"):
                        video_file = os.path.join(subdir_path, file_name)
                        base_name = file_name.replace(".mkv", "")
                        pitch_file = os.path.join(self.base_dir, "pitch", f"{subdir}-pitch-{base_name}.npy")
                        energy_file = os.path.join(self.base_dir, "energy", f"{subdir}-energy-{base_name}.npy")
                        duration_file = os.path.join(self.base_dir, "duration", f"{subdir}-duration-{base_name}.npy")
                        mel_file = os.path.join(self.base_dir, "mel", f"{subdir}-mel-{base_name}.npy")
                        
                        if all(os.path.exists(f) for f in [video_file, pitch_file, energy_file, duration_file, mel_file]):
                            self.video_files.append(video_file)
                            self.feature_files[video_file] = {
                                "pitch": pitch_file,
                                "energy": energy_file,
                                "duration": duration_file,
                                "mel": mel_file,
                            }

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        feature_paths = self.feature_files[video_file]

        pitch = np.load(feature_paths["pitch"])
        energy = np.load(feature_paths["energy"])
        duration = np.load(feature_paths["duration"])
        mel = np.load(feature_paths["mel"])
        duration = get_video_duration(video_file)
        num_frames = decide_num_frames(video_file, duration) ##1s
        #print("duration",duration)
        #print(num_frames)
        self.num_frames = num_frames
        #video_data = preprocess_video(video_file, self.feature_extractor, self.num_frames)
        video_data = preprocess_video(video_file, self.feature_extractor, num_frames=10)

        if self.target_type == 'pitch':
            target = pitch
        elif self.target_type == 'energy':
            target = energy
        elif self.target_type == 'duration':
            target = duration
        else:
            raise ValueError("Invalid target_type. Must be 'pitch', 'energy', or 'duration'.")

        return video_data, target, video_file

class ViTPitchModel(nn.Module):
    def __init__(self, output_size):
        super(ViTPitchModel, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.fc1 = nn.Linear(self.feature_extractor.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state[:, 0, :]
        x = x.view(batch_size, num_frames, -1).mean(dim=1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ViTPitchModel2(nn.Module):
    def __init__(self, output_size):
        super(ViTPitchModel2, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.lstm = nn.LSTM(self.feature_extractor.config.hidden_size, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128*2, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state[:, 0, :]
        x = x.view(batch_size, num_frames, -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ViTPitchModel3(nn.Module):
    def __init__(self, output_size):
        super(ViTPitchModel3, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.fc1 = nn.Linear(self.feature_extractor.config.hidden_size, 64)
        #self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state[:, 0, :]
        print("0",x.shape)
        x = x.view(batch_size, num_frames, -1).mean(dim=1)
        print("1",x.shape)
        x = F.relu(self.fc1(x))
        print("2",x.shape)
        #x = F.relu(self.fc2(x))
        print("3",x.shape)
        x = self.fc3(x)
        print("4",x.shape)
        return x


##add model
#CNN+Attention+LSTM, ViT
class CNNAttentionModel(nn.Module):
    def __init__(self, output_size):
        super(CNNAttentionModel, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # Adjusted for bidirectional LSTM output size
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, output_size)  # Adjusted for bidirectional LSTM output size

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        x, _ = self.lstm(x)
        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        x = attn_output.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


#CNN+Attention, ViT
class CNNAttentionModel1(nn.Module):
    def __init__(self, output_size):
        super(CNNAttentionModel1, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)##
        
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        print("0",x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        
        print("1",x.shape)
        #x = x.mean(dim=2)  # [batch_size * num_frames, 64]
        
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, hidden_size]
        print("2",x.shape)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.mean(dim=0)  # [batch_size, hidden_size]
        print("3",attn_output.shape)
        x = attn_output.view(batch_size, num_frames, -1).mean(dim=1)
        print("33",x.shape)
        x = self.fc(x)
        print("4",x.shape)
        return x

class CNNAttentionModel2(nn.Module):
    def __init__(self, output_size):
        super(CNNAttentionModel2, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)##
        
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        print("0",x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        
        print("1",x.shape)
        #x = x.mean(dim=2)  # [batch_size * num_frames, 64]
        
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, hidden_size]
        print("2",x.shape)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.mean(dim=0)  # [batch_size, hidden_size]
        print("3",attn_output.shape)
        x = attn_output.view(batch_size, num_frames, -1).mean(dim=1)
        print("33",x.shape)
        x = self.fc(x)
        print("4",x.shape)
        return x


class ConvModel(nn.Module):
    def __init__(self, output_size):
        super(ConvModel, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, output_size)  # Adjusted for bidirectional LSTM output size

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        x, _ = self.lstm(x)
        
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class ConvModel2(nn.Module):
    def __init__(self, output_size):
        super(ConvModel, self).__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        outputs = self.feature_extractor(pixel_values=x, output_hidden_states=True)
        x = outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]
        x = x.permute(0, 2, 1)  # [batch_size * num_frames, hidden_size, seq_len]
        #print("1",x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #print("2",x.shape)
        x = x.mean(dim=2)  # [batch_size * num_frames, 64]
        #print("3",x.shape)
        x = x.view(batch_size, num_frames, -1).mean(dim=1)
        #print("33",x.shape)
        x = self.fc(x)
        #print("4",x.shape)
        return x

#CNN+Attention+LSTM, ConvNext    
class ConvNeXtModel(nn.Module):
    def __init__(self, output_size):
        super(ConvNeXtModel, self).__init__()
        self.feature_extractor = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.num_features, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Adjusted for bidirectional LSTM output size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.feature_extractor(x)
        x = features.view(batch_size, num_frames, -1)
        
        # Apply CNN layers
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Apply LSTM layer
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        x, _ = self.lstm(x)
        
        x = x.mean(dim=1)
        
        # Apply Dropout layer
        x = self.dropout(x)
        
        # Apply Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
#CNN+Attention+LSTM, SWIN    
class CNNAttentionModel_SWIN(nn.Module):
    def __init__(self, output_size):
        super(CNNAttentionModel_SWIN, self).__init__()
        self.feature_extractor = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.num_features, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # Adjusted for bidirectional LSTM output size
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, output_size)  # Adjusted for bidirectional LSTM output size

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        
        features = self.feature_extractor.forward_features(x)
        features = features.mean(dim=[1, 2])  # Global average pooling to reduce 7x7 to 1
        
        x = features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        x, _ = self.lstm(x)
        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        x = attn_output.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    
class CNNAttentionModel_SWIN2(nn.Module):
    def __init__(self, output_size):
        super(CNNAttentionModel_SWIN, self).__init__()
        self.feature_extractor = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.conv1 = nn.Conv1d(in_channels=self.feature_extractor.num_features, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        
        features = self.feature_extractor.forward_features(x)
        print("features shape:", features.shape)
        
        # Global average pooling to reduce 7x7 to 1
        features = features.mean(dim=[1, 2])
        print("features after pooling:", features.shape)
        
        x = features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        print("0", x.shape)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        print("1", x.shape)
        
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, hidden_size]
        print("2", x.shape)
        
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.mean(dim=0)  # [batch_size, hidden_size]
        print("3", attn_output.shape)
        
        x = attn_output.view(batch_size, -1)  # [batch_size, hidden_size]
        print("33", x.shape)
        
        x = self.fc(x)
        print("4", x.shape)
        
        return x    


def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames  # Calculate interval between frames
    frame_indices = [i * interval for i in range(num_frames)]

    extracted_frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_idx}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224)
        extracted_frames.append(frame)

    cap.release()
    return extracted_frames



def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def decide_num_frames(video_path, duration_seconds):
    frame_rate = get_frame_rate(video_path)
    num_frames = int(frame_rate * duration_seconds)
    return num_frames

def pad_or_truncate(tensor, target_length):
    length = tensor.size(1)
    if length > target_length:
        return tensor[:, :target_length]
    elif length < target_length:
        pad_size = target_length - length
        return F.pad(tensor, (0, 0, 0, pad_size), mode='constant', value=0)
    else:
        return tensor

def pad_collate(batch):
    videos, targets, video_files = zip(*batch)
    max_length = max(video.shape[1] for video in videos)
    videos_padded = pad_sequence([video.clone().detach() for video in videos], batch_first=True, padding_value=0)
    targets_padded = pad_sequence([torch.tensor(target).clone().detach() for target in targets], batch_first=True, padding_value=0)
    return videos_padded, targets_padded, video_files

def create_mask(tensor, padding_value=0):
    # Create a mask for non-padding elements
    mask = tensor != padding_value
    return mask

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, start_epoch=0, start_step=0):
    model.train()
    step = start_step
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for batch_idx, (video_data, targets, video_files) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(video_data)
            targets = torch.tensor(targets).float().clone().detach().to(outputs.device)
            
            # Ensure both tensors have the same length
            max_length = max(outputs.size(1), targets.size(1))
            outputs = pad_or_truncate(outputs.unsqueeze(0), max_length).squeeze(0)
            targets = pad_or_truncate(targets.unsqueeze(0), max_length).squeeze(0)
            #print("1")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            # Create masks
            output_mask = create_mask(outputs)
            target_mask = create_mask(targets)
            #print("2 before mask")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            # Apply masks
            outputs = outputs.masked_select(output_mask).view(-1)
            targets = targets.masked_select(target_mask).view(-1)
            
            #print("3 after mask mask")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            
            if outputs.size(0) != targets.size(0):
                min_length = min(outputs.size(0), targets.size(0))
                outputs = outputs[:min_length]
                targets = targets[:min_length]
                
            #print("4 after tuning size ")    
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
            # Save checkpoint every 100 steps
            if step % 5000 == 0:
                save_checkpoint(model, optimizer, epoch, step, "{}_SWIN.pth.tar".format(step))
                
            step += 1

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}')

def save_checkpoint(model, optimizer, epoch, step, filename):
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch}, step {step}")

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        step = state['step']
        print(f"Checkpoint loaded from epoch {epoch}, step {step}")
        return epoch, step
    else:
        print("No checkpoint found.")
        return 0, 0



def train_model2(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (video_data, targets, video_files) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(video_data)
            targets = torch.tensor(targets).float().clone().detach().to(outputs.device)
            
            # Ensure both tensors have the same length
            max_length = max(outputs.size(1), targets.size(1))
            outputs = pad_or_truncate(outputs.unsqueeze(0), max_length).squeeze(0)
            targets = pad_or_truncate(targets.unsqueeze(0), max_length).squeeze(0)
            #print("1")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            # Create masks
            output_mask = create_mask(outputs)
            target_mask = create_mask(targets)
            #print("2 before mask")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            # Apply masks
            outputs = outputs.masked_select(output_mask).view(-1)
            targets = targets.masked_select(target_mask).view(-1)
            
            #print("3 after mask mask")
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            
            if outputs.size(0) != targets.size(0):
                min_length = min(outputs.size(0), targets.size(0))
                outputs = outputs[:min_length]
                targets = targets[:min_length]
                
            #print("4 after tuning size ")    
            #print(outputs,outputs.shape)
            #print(targets,targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}')

def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    all_video_files = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (video_data, targets, video_file) in enumerate(data_loader):
            outputs = model(video_data)
            targets = torch.tensor(targets).float().clone().detach().to(outputs.device)
            
            # Ensure both tensors have the same length
            max_length = max(outputs.size(1), targets.size(1))
            outputs = pad_or_truncate(outputs.unsqueeze(0), max_length).squeeze(0)
            targets = pad_or_truncate(targets.unsqueeze(0), max_length).squeeze(0)
            
            # Create masks
            output_mask = create_mask(outputs)
            target_mask = create_mask(targets)
            
            # Apply masks
            outputs = outputs.masked_select(output_mask).view(-1)
            targets = targets.masked_select(target_mask).view(-1)
            
            if outputs.size(0) != targets.size(0):
                min_length = min(outputs.size(0), targets.size(0))
                outputs = outputs[:min_length]
                targets = targets[:min_length]

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_video_files.append(video_file)

    rmse = np.sqrt(mean_squared_error(np.concatenate(all_targets), np.concatenate(all_preds)))
    avg_loss = total_loss / len(data_loader)
    return rmse, avg_loss, all_preds, all_targets, all_video_files

def preprocess_video2(video_path, feature_extractor, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224)
        frames.append(frame)
        frame_count += 1
        if num_frames and frame_count >= num_frames:
            break
    cap.release()
    encoding = feature_extractor(images=frames, return_tensors="pt")
    video_data = encoding['pixel_values']
    return video_data


def preprocess_video(video_path, feature_extractor, num_frames=5):
    frames = extract_frames(video_path, num_frames=num_frames)
    if not frames:
        return torch.empty(0)
    encoding = feature_extractor(images=frames, return_tensors="pt")
    video_data = encoding['pixel_values']
    return video_data


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


#plot function
def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


# Function to plot pitch and energy contours
def plot_pitch_energy2(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        #energy = energy * energy_std + energy_mean
        
        ax = axes[i][0]
        ax.plot(pitch, color="tomato", label="Pitch")
        ax.set_ylim(pitch_min, pitch_max)
        ax.set_ylabel("Pitch (Hz)", color="tomato")
        ax.tick_params(axis='y', labelcolor="tomato")

        ax2 = ax.twinx()
        ax2.plot(energy, color="darkviolet", label="Energy")
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy (dB or arbitrary units)", color="darkviolet")
        ax2.tick_params(axis='y', labelcolor="darkviolet")

        ax.set_title(titles[i])

    fig.tight_layout()
    return fig



# Function to plot pitch and energy contours with predictions
def plot_pitch_energy_with_predictions(data, stats, titles, predictions=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    for i in range(len(data)):
        pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        ax = axes[i][0]
        ax.plot(pitch, color="tomato", label="Actual Pitch")
        ax.set_ylim(pitch_min, pitch_max)
        ax.set_ylabel("Pitch (Hz)", color="tomato")
        ax.tick_params(axis='y', labelcolor="tomato")

        ax2 = ax.twinx()
        ax2.plot(energy, color="darkviolet", label="Actual Energy")
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy (dB or arbitrary units)", color="darkviolet")
        ax2.tick_params(axis='y', labelcolor="darkviolet")

        if predictions:
            pred_pitch, pred_energy = predictions[i]
            pred_pitch = pred_pitch * pitch_std + pitch_mean
            ax.plot(pred_pitch, color="blue", linestyle="dashed", label="Predicted Pitch")
            ax2.plot(pred_energy, color="green", linestyle="dashed", label="Predicted Energy")

        ax.set_title(titles[i])

    fig.tight_layout()
    fig.legend(loc='upper right')
    return fig

# Function to compare and plot actual and predicted pitch and energy
#def compare_and_plot_predictions(actual_data, predicted_data, stats, titles):
#    fig = plot_pitch_energy_with_predictions(actual_data, stats, titles, predictions=predicted_data)
#    return fig

def get_predicted_data(all_preds, all_targets, idx):
    actual_pitch = all_targets[idx] #.flatten()
    predicted_pitch = all_preds[idx] #.flatten()
    return actual_pitch, predicted_pitch


# Function to plot pitch contours with predictions
def plot_pitch_with_predictions(actual_pitch, predicted_pitch, stats, title):
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    fig, ax = plt.subplots()
    actual_pitch = actual_pitch * pitch_std + pitch_mean
    predicted_pitch = predicted_pitch * pitch_std + pitch_mean

    ax.plot(actual_pitch, color="tomato", label="Actual Pitch")
    ax.plot(predicted_pitch, color="blue", linestyle="dashed", label="Predicted Pitch")
    ax.set_ylim(pitch_min, pitch_max)
    ax.set_ylabel("Pitch (Hz)", color="tomato")
    ax.set_title(title)
    ax.legend(loc='upper right')

    return fig

# Function to compare and plot actual and predicted pitch
def compare_and_plot_pitch_predictions(all_preds, all_targets, stats, video_files, idx):
    actual_pitch, predicted_pitch = get_predicted_data(all_preds, all_targets, idx)
    title = f"Sample {idx} - {os.path.basename(video_files[idx])}"
    fig = plot_pitch_with_predictions(actual_pitch, predicted_pitch, stats, title)
    return fig

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    #print("frame_rate",frame_rate)
    duration = frame_count / frame_rate
    cap.release()
    return duration

def compute_video_durations(video_files):
    durations = []
    for video_file in video_files:
        duration = get_video_duration(video_file)
        durations.append(duration)
    return durations

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def compute_wav_durations(rawfile_dir):
    durations = []
    for subdir in os.listdir(rawfile_dir):
        subdir_path = os.path.join(rawfile_dir, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".wav"):
                    wav_path = os.path.join(subdir_path, file_name)
                    duration = get_wav_duration(wav_path)
                    durations.append(duration)
                    #print(f"{file_name}: Duration = {duration:.2f} seconds")
    return durations

def print_statistics(durations, file_type):
    if durations:
        total_duration = np.sum(durations)
        mean_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        total_hours = total_duration / 3600  # Convert to hours
        print(f'\nTotal duration of {file_type} files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')
        print(f'Mean duration of {file_type} files: {mean_duration:.2f} seconds')
        print(f'Min duration of {file_type} files: {min_duration:.2f} seconds')
        print(f'Max duration of {file_type} files: {max_duration:.2f} seconds')
        logging.info(f'Total duration of {file_type} files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')
        logging.info(f'Mean duration of {file_type} files: {mean_duration:.2f} seconds')
        logging.info(f'Min duration of {file_type} files: {min_duration:.2f} seconds')
        logging.info(f'Max duration of {file_type} files: {max_duration:.2f} seconds')
    else:
        print(f'No {file_type} files found.')

if __name__ == "__main__":
    logging.info("Start")
    base_dir = '/Users/khs/Downloads/dissertation/reference_works/hyunsik/visualprosody/CMD-1s-50-full/CMD-1s-50-test'
    target_type = 'pitch'
    num_frames = 32

    dataset = VideoPitchDataset(base_dir, target_type=target_type, num_frames=num_frames)
    rawfile_dir = base_dir + "/rawfile"
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the file paths and ensure that all required files are present.")
    '''
    total_duration = compute_total_video_duration(dataset.video_files)
    total_hours = total_duration / 3600  # Convert to hours
    print(f'Total duration of video files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')
    logging.info(f'Total duration of video files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')

    
    total_duration, total_hours = compute_total_wav_duration(rawfile_dir)
    logging.info(f'Total duration of WAV files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')    
    '''

    # Compute and print WAV file durations
    wav_durations = compute_wav_durations(rawfile_dir)
    print_statistics(wav_durations, "WAV")

    # Collect MKV file paths
    mkv_files = []
    for subdir in os.listdir(rawfile_dir):
        subdir_path = os.path.join(rawfile_dir, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".mkv"):
                    mkv_files.append(os.path.join(subdir_path, file_name))

    # Compute and print MKV file durations
    video_durations = compute_video_durations(mkv_files)
    print_statistics(video_durations, "MKV")


        # Split the dataset into training and evaluation sets
    train_indices, eval_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    print(eval_indices)
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    
    
    #train_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    sample_target = dataset[0][1]
    if isinstance(sample_target, np.ndarray):
        output_size = len(sample_target)
    else:
        output_size = 1

    output_size = 20
    print("output_size", output_size)
    
    #Basic
    #logging.info("ViTPitchModel")
    #model_name = "ViTPitchModel"
    #print("ViT")
    #model = ViTPitchModel(output_size=output_size)
    
    #logging.info("ConvModel")
    #model_name = "CNN"
    #print("ConvModel")
    #model = ConvModel(output_size=output_size)
    
    
    #CNN+Attention
    #logging.info("CNNAttentionModel1")
    #model_name = "CA1"
    #print("CNNAttentionModel1")
    #model = CNNAttentionModel1(output_size=output_size)

    #CNN+Attention+LSTM
    logging.info("CNNAttentionModel")
    model_name = "CALSTM"
    print("CNNAttentionModel")
    model = CNNAttentionModel(output_size=output_size)
    
    #logging.info("CNNAttentionModel_SWIN")
    #model_name = "SWIN"
    #print("CNNAttentionModel_SWIN")
    #model = CNNAttentionModel_SWIN(output_size=output_size)
    
    #logging.info("ConvNeXtModel")
    #model_name = "CNeX"
    #print("ConvNeXtModel")
    #5model = ConvNeXtModel(output_size=output_size)
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    
    # Check for existing checkpoint
    start_epoch, start_step = load_checkpoint(model, optimizer, '100.pth.tar')

    train_model(model, train_loader, criterion, optimizer, num_epochs=3, start_epoch=start_epoch, start_step=start_step)
    torch.save(model.state_dict(), f'vit_pitch_model_{model_name}.pth')
    print("=====train ended and saved=======")

    print("=====load trained model======")
    model.load_state_dict(torch.load(f'vit_pitch_model_{model_name}.pth'))

    model.eval()
    print("=======evaluate start=========")
    rmse, avg_loss, all_preds, all_targets, all_video_files = evaluate_model(model, eval_loader, criterion)
    logging.info(f'RMSE: {rmse:.4f}, Avg Loss: {avg_loss:.4f}')
    print(f'RMSE: {rmse:.4f}, Avg Loss: {avg_loss:.4f}')

    
    # Compute RMSE for each sample
    sample_rmses = []
    for pred, actual, video_file in zip(all_preds, all_targets, all_video_files):
        sample_rmse = np.sqrt(mean_squared_error(actual.flatten(), pred.flatten()))
        sample_rmses.append((sample_rmse, video_file, pred, actual))
    
    # Sort by RMSE and get top 5 best RMSE samples
    top_5_rmses = sorted(sample_rmses, key=lambda x: x[0])[:5]
    
    # Print top 5 best RMSE samples
    print("Top 5 best RMSE samples:")
    for rmse, video_file, pred, actual in top_5_rmses:
        video_file_name = os.path.basename(video_file[0])  # Extract the video file name
        logging.info(f'{video_file_name}: RMSE = {rmse:.4f}, Actual {target_type.capitalize()} = {actual.flatten()[:10]}, Predicted {target_type.capitalize()} = {pred.flatten()[:10]}')
        print(f'{video_file_name}: RMSE = {rmse:.4f}, Actual {target_type.capitalize()} = {actual.flatten()[:10]}, Predicted {target_type.capitalize()} = {pred.flatten()[:10]}')
    
    
    
    # Print comparison between actual and predicted mean pitch or energy
    for i, (pred, actual, video_file) in enumerate(zip(all_preds, all_targets, all_video_files)):
        video_file_name = os.path.basename(video_file[0])  # Extract the video file name
        logging.info(f'{video_file_name}: Actual {target_type.capitalize()} = {actual.flatten()[:10]}, Predicted {target_type.capitalize()} = {pred.flatten()[:10]}')
        print(f'{video_file_name}: Actual {target_type.capitalize()} = {actual.flatten()[:10]}, Predicted {target_type.capitalize()} = {pred.flatten()[:10]}')
        
        
        
    # Load stats data from stats.json
    with open(base_dir + '/stats.json', 'r') as f:
        stats = json.load(f)
        
    # Load stats data from stats.json 
    #test sample -LJSpeech
    #with open('/Users/khs/Downloads/dissertation/reference_works/hyunsik/visualprosody/LJSpeech/stats.json', 'r') as f:
    #    stats = json.load(f)
        
    
    pitch_stats = stats["pitch"]
    energy_stats = stats["energy"]
    stats2 = stats["pitch"] + stats["energy"][:2]
    
    pitch_min, pitch_max, pitch_mean, pitch_std = pitch_stats
    energy_min, energy_max, energy_mean, energy_std = energy_stats


    # Plot pitch and energy contours
    for i, (video_data, targets, video_file) in enumerate(eval_loader):
        eval_idx = eval_indices[i]  # Get the corresponding index in the original dataset
        pitch = np.load(dataset.feature_files[dataset.video_files[eval_idx]]["pitch"])
        energy = np.load(dataset.feature_files[dataset.video_files[eval_idx]]["energy"])
        duration = np.load(dataset.feature_files[dataset.video_files[eval_idx]]["duration"])
        mel = np.load(dataset.feature_files[dataset.video_files[eval_idx]]["mel"])
    
        pitch = expand(pitch, duration)
        energy = expand(energy, duration)
        mel = mel.T
        fig = plot_mel([
                (mel, pitch, energy),
            ],
            stats2,
            ["Synthetized Spectrogram"],)
        video_base_name = os.path.basename(dataset.video_files[eval_idx]).replace(".mkv", "")
        print("video_base_name", video_base_name)
        fig.savefig(f'{video_base_name}_{model_name}_plot_mel.png')
        plt.close(fig)
    
        fig = plot_pitch_energy2([(pitch, energy)], stats2, [f'Sample {i}'],)
        fig.savefig(f'{video_base_name}_{model_name}_sample_plot_pitch_energy.png')
        plt.close(fig)
       
        
        
    # Plot pitch contours with predictions
    for i, (video_data, targets, video_file) in enumerate(eval_loader):
        eval_idx = eval_indices[i]  # Get the corresponding index in the original dataset
        print(eval_indices)
        print(eval_idx)
        video_base_name = os.path.basename(dataset.video_files[eval_idx]).replace(".mkv", "")
        fig = compare_and_plot_pitch_predictions(all_preds, all_targets, stats2, [dataset.video_files[idx] for idx in eval_indices], i)
        fig.savefig(f'{video_base_name}_{model_name}_comparison_plot_pitch.png')
        plt.close(fig)     
        
        
        
        
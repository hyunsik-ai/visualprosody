## change log
#1. Add function to save preprocessed video
#2. modify video features extraction part(cls and patch all considered)
#3. save video features as pt file, and load when call next time.

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor, ViTModel
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm


# Configure logging
logging_model_name = "LSTM"  # or any other model name
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{logging_model_name}_training_visualprosody.log'),
        logging.StreamHandler()  # Log to console as well
    ]
)

class VideoPitchDataset(Dataset):
    def __init__(self, base_dir, target_type='pitch', num_frames_per_target=2):
        self.base_dir = base_dir
        self.rawfile_dir = base_dir + "/rawfile"
        self.target_type = target_type
        self.video_files = []
        self.feature_files = {}
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.num_frames_per_target = num_frames_per_target
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
                        feature_file = os.path.join(self.base_dir, "feature", f"{subdir}-feature-{base_name}.pt")
                        if os.path.exists(video_file) and os.path.exists(pitch_file):
                            self.video_files.append(video_file)
                            self.feature_files[video_file] = {
                                "pitch": pitch_file,
                                "duration": duration_file,
                                "feature": feature_file
                            }

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        pitch = np.load(self.feature_files[video_file]["pitch"])
        duration = np.load(self.feature_files[video_file]["duration"])
        feature_file = self.feature_files[video_file]["feature"]

        if os.path.exists(feature_file):
            features = torch.load(feature_file)
            print(f"Loaded precomputed features from {feature_file}")
            logging.info(f"Loaded precomputed features from {feature_file}")
        else:
            # Extract features if not already saved
            video_data = self.preprocess_video(video_file)
            features = self.extract_features(video_data)
            torch.save(features, feature_file)
            logging.info(f"Extracted and saved features to {feature_file}")

        # Assign features based on duration
        grouped_features = self.assign_features(features, duration)
        logging.info(f"Grouped features into {len(grouped_features)} groups based on duration.")
        print(f"Grouped features into {len(grouped_features)} groups based on duration.")

        # Pad or truncate features to match number of pitch values
        num_pitch_values = len(pitch)
        grouped_features = self.pad_or_truncate_features(grouped_features, num_pitch_values)

        return grouped_features, torch.tensor(pitch, dtype=torch.float32)

    # ... other methods remain unchanged ...

    def preprocess_video(self, video_path):
        frames = self.extract_frames(video_path, self.num_frames_per_target * len(np.load(self.feature_files[video_path]["pitch"])))
        if not frames:
            return torch.empty(0)
        encoding = self.feature_extractor(images=frames, return_tensors="pt")
        return encoding['pixel_values']

    def extract_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // num_frames, 1)  # Calculate interval between frames
        frame_indices = [i * interval for i in range(num_frames)]

        extracted_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Error: Could not read frame {frame_idx} in video {video_path}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224)
            extracted_frames.append(frame)

        cap.release()
        
        # Save the processed frames as a video to verify preprocessing
        output_video_path = video_path.replace('.mkv', '_processed.mkv')
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
        for frame in extracted_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        logging.info(f"Processed frames saved to {output_video_path}")
        return extracted_frames

    def extract_features(self, video_data):
        with torch.no_grad():
            outputs = self.vit_model(pixel_values=video_data)
            hidden_states = outputs.last_hidden_state  # [num_frames, num_patches, hidden_size]
            #Processing Patch Embeddings
            features = hidden_states.max(dim=1)[0]  # Max pooling [num_frames, hidden_size]
            #features = hidden_states.mean(dim=1)  # Average pooling [num_frames, hidden_size]
            #features = hidden_states[:, 0, :]  # Use the CLS token representation
        return features

    def assign_features(self, features, duration):
        total_duration = np.sum(duration)
        grouped_features = []
        print("total_duration",total_duration)
        feature_idx = 0
        for dur in duration:
            num_features = int(len(features) * (dur / total_duration))
            for _ in range(num_features):
                if feature_idx < len(features):
                    grouped_features.append(features[feature_idx])
                    feature_idx += 1
        print("len(grouped_features)1",len(grouped_features))
        if len(grouped_features) < len(duration):
            padding = torch.zeros((len(duration) - len(grouped_features), features.shape[1]))
            grouped_features.extend(padding)
        print("len(grouped_features)",len(grouped_features))
        print("len(duration)",len(duration))
        return torch.stack(grouped_features[:len(duration)])

    def pad_or_truncate_features(self, features, target_length):
        print("target_length",target_length)
        print("len(features)1",len(features))
        if len(features) > target_length:
            return features[:target_length]
        elif len(features) < target_length:
            padding = torch.zeros((target_length - len(features), features.shape[1]))
            return torch.cat([features, padding], dim=0)
        
        print("len(features)",len(features))
        return features
    

class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class CNNModel(nn.Module):
    def __init__(self, input_size=768, num_channels=128, output_size=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_channels // 2, num_channels // 4)
        self.fc2 = nn.Linear(num_channels // 4, output_size)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, feature_size, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, num_channels // 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FFNNModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        x = x.view(batch_size * seq_len, feature_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(batch_size, seq_len)
        return x
    
def evaluate_model_FFNN(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs)
            targets = targets.view(outputs.shape)  # Ensure targets have the same shape as outputs
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Debug prints to check shapes
            logging.info(f"Batch {batch_idx} - outputs shape: {outputs.shape}, targets shape: {targets.shape}")

    # Padding or truncating sequences to the same length for concatenation
    max_length = max([pred.shape[1] for pred in all_preds])

    def pad_sequence(seq, max_length):
        pad_width = max_length - seq.shape[1]
        if pad_width > 0:
            return np.pad(seq, ((0, 0), (0, pad_width)), 'constant')
        return seq

    all_preds_padded = [pad_sequence(pred, max_length) for pred in all_preds]
    all_targets_padded = [pad_sequence(target, max_length) for target in all_targets]

    avg_loss = total_loss / len(data_loader)
    return np.concatenate(all_preds_padded), np.concatenate(all_targets_padded), avg_loss

def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs)
            targets = targets.view(outputs.shape)  # Ensure targets have the same shape as outputs
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Debug prints to check shapes
            logging.info(f"Batch {batch_idx} - outputs shape: {outputs.shape}, targets shape: {targets.shape}")

    # Padding or truncating sequences to the same length for concatenation
    max_length = max([pred.shape[1] for pred in all_preds])

    def pad_sequence(seq, max_length):
        pad_width = max_length - seq.shape[1]
        if pad_width > 0:
            return np.pad(seq, ((0, 0), (0, pad_width), (0, 0)), 'constant')
        return seq

    all_preds_padded = [pad_sequence(pred, max_length) for pred in all_preds]
    all_targets_padded = [pad_sequence(target, max_length) for target in all_targets]

    avg_loss = total_loss / len(data_loader)
    return np.concatenate(all_preds_padded), np.concatenate(all_targets_padded), avg_loss

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        eval_losses = checkpoint['eval_losses']
        step = checkpoint['step']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    else:
        logging.info(f"No checkpoint found at '{checkpoint_path}'")
        start_epoch = 0
        train_losses = []
        eval_losses = []
        step = 0

    return model, optimizer, start_epoch, train_losses, eval_losses, step

def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved at '{checkpoint_path}'")

# Initialize the dataset and dataloader
base_dir = '/Users/khs/Downloads/dissertation/reference_works/hyunsik/visualprosody/CMD-1s-50-full/CMD-1s-50-test'
dataset = VideoPitchDataset(base_dir)

# Split dataset
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
model = LSTMModel() # LSTMModel() #CNNModel() #FFNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
eval_losses = []

start_epoch = 0
checkpoint_path = 'checkpoint_step_20.pth'
model, optimizer, start_epoch, train_losses, eval_losses, step = load_checkpoint(checkpoint_path, model, optimizer)

# Load from last model if available
logging.info("=====load last trained model======")
# model.load_state_dict(torch.load('last_save_trainmodel.pth'))     

# Training loop with checkpoint saving every 1000 steps
num_epochs = 5
checkpoint_interval = 500
step = 0

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.view(outputs.shape)  # Ensure targets have the same shape as outputs
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1

        if batch_idx % 5 == 0:
            logging.info(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if step % checkpoint_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'eval_losses': eval_losses,
                'step': step,
            }, f'checkpoint_step_{step}_{logging_model_name}.pth')

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    logging.info(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}] Loss: {epoch_loss:.4f}')

    model.eval()
    predictions, targets, eval_loss = evaluate_model(model, eval_loader, criterion)  # CNN, LSTM
    # predictions, targets, eval_loss = evaluate_model_FFNN(model, eval_loader, criterion)  # FFNN
    logging.info(f"Predictions: {predictions}")
    logging.info(f"Targets: {targets}")
    eval_losses.append(eval_loss)
    logging.info(f'Evaluation Loss after epoch {epoch + 1}: {eval_loss:.4f}')

# Plot training and evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Evaluation Loss')
plt.savefig(f'training_evaluation_loss{logging_model_name}.png')  # Save the plot as an image file

# Plot predictions vs. targets
plt.figure(figsize=(10, 5))
plt.plot(predictions.flatten(), label='Predictions')
plt.plot(targets.flatten(), label='Targets')
plt.xlabel('Time')
plt.ylabel('Pitch')
plt.legend()
plt.title('Predictions vs. Targets')
plt.savefig(f'predictions_vs_targets{logging_model_name}.png')  # Save the plot as an image file

plt.figure(figsize=(10, 5))
plt.plot(predictions.flatten(), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Pitch')
plt.legend()
plt.title('Predictions')
plt.savefig(f'predictions{logging_model_name}.png')  # Save the plot as an image file

plt.figure(figsize=(10, 5))
plt.plot(targets.flatten(), label='Targets')
plt.xlabel('Time')
plt.ylabel('Pitch')
plt.legend()
plt.title('Targets')
plt.savefig(f'targets{logging_model_name}.png')  # Save the plot as an image file

torch.save(model.state_dict(), f'last_save_trainmodel_{logging_model_name}.pth')

# Evaluate the model
predictions, targets, eval_loss = evaluate_model(model, eval_loader, criterion)
logging.info(f'Evaluation Loss: {eval_loss:.4f}')
# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - targets) ** 2))
logging.info(f'RMSE: {rmse:.4f}')

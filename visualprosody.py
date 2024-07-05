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
                        if os.path.exists(video_file) and os.path.exists(pitch_file):
                            self.video_files.append(video_file)
                            self.feature_files[video_file] = {"pitch": pitch_file, "duration": duration_file}

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        pitch = np.load(self.feature_files[video_file]["pitch"])
        duration = np.load(self.feature_files[video_file]["duration"])

        # Extract features
        video_data = self.preprocess_video(video_file)
        features = self.extract_features(video_data)
        print("features",len(features))

        # Assign features based on duration
        grouped_features = self.assign_features(features, duration)
        print("grouped_features",len(grouped_features))

        # Pad or truncate features to match number of pitch values
        num_pitch_values = len(pitch)
        grouped_features = self.pad_or_truncate_features(grouped_features, num_pitch_values)
        
        return grouped_features, torch.tensor(pitch, dtype=torch.float32)

    def preprocess_video(self, video_path):
        frames = self.extract_frames(video_path, self.num_frames_per_target * len(np.load(self.feature_files[video_path]["pitch"])))
        if not frames:
            return torch.empty(0)
        encoding = self.feature_extractor(images=frames, return_tensors="pt")
        return encoding['pixel_values']

    def extract_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // num_frames, 1)  # Calculate interval between frames
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

    def extract_features(self, video_data):
        with torch.no_grad():
            outputs = self.vit_model(pixel_values=video_data)
            hidden_states = outputs.last_hidden_state  # [num_frames, num_patches, hidden_size]
            features = hidden_states[:, 0, :]  # Use the CLS token representation
        return features

    def assign_features(self, features, duration):
        total_duration = np.sum(duration)
        grouped_features = []

        feature_idx = 0
        for dur in duration:
            num_features = int(len(features) * (dur / total_duration))
            for _ in range(num_features):
                if feature_idx < len(features):
                    grouped_features.append(features[feature_idx])
                    feature_idx += 1

        if len(grouped_features) < len(duration):
            padding = torch.zeros((len(duration) - len(grouped_features), features.shape[1]))
            grouped_features.extend(padding)

        return torch.stack(grouped_features[:len(duration)])

    def pad_or_truncate_features(self, features, target_length):
        if len(features) > target_length:
            return features[:target_length]
        elif len(features) < target_length:
            padding = torch.zeros((target_length - len(features), features.shape[1]))
            return torch.cat([features, padding], dim=0)
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
            print(f"Batch {batch_idx} - outputs shape: {outputs.shape}, targets shape: {targets.shape}")

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
            #print("outputs element",outputs)
            #print("targets element",targets)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Debug prints to check shapes
            print(f"Batch {batch_idx} - outputs shape: {outputs.shape}, targets shape: {targets.shape}")

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
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        eval_losses = checkpoint['eval_losses']
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        start_epoch = 0
        train_losses = []
        eval_losses = []

    return model, optimizer, start_epoch, train_losses, eval_losses


def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}'")


# Initialize the dataset and dataloader
base_dir = '/Users/khs/Downloads/dissertation/reference_works/hyunsik/visualprosody/CMD-1s-50-full/CMD-1s-50'
dataset = VideoPitchDataset(base_dir)

# Split dataset
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNModel() # LSTMModel() #CNNModel() #FFNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
eval_losses = []


start_epoch = 0
# Load from checkpoint if available
#print("check point loaded")
#checkpoint_path = 'checkpoint_step_20.pth'
#model, optimizer, start_epoch, train_losses, eval_losses = load_checkpoint(checkpoint_path, model, optimizer)


# Load from last model if available
print("=====load last trained model======")
model.load_state_dict(torch.load('last_save_trainmodel.pth'))     

# Training loop with checkpoint saving every 1000 steps
num_epochs = 5
checkpoint_interval = 100
step = 0

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0
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
            print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if step % checkpoint_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'eval_losses': eval_losses,
            }, f'checkpoint_step_{step}.pth')

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}] Loss: {epoch_loss:.4f}')

    model.eval()
    #predictions, targets, eval_loss = evaluate_model_FFNN(model, eval_loader, criterion) #FFNN
    predictions, targets, eval_loss = evaluate_model(model, eval_loader, criterion) ##CNN, LSTM
    print("predictions",predictions)
    print("targets",targets)
    eval_losses.append(eval_loss)
    print(f'Evaluation Loss after epoch {epoch + 1}: {eval_loss:.4f}')

    # Save the checkpoint at the end of each epoch
    '''
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'eval_losses': eval_losses,
    }, checkpoint_path)
    '''

# Plot training and evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Evaluation Loss')
plt.savefig('training_evaluation_loss.png')  # Save the plot as an image file

# Plot predictions vs. targets
plt.figure(figsize=(10, 5))
plt.plot(predictions.flatten(), label='Predictions')
plt.plot(targets.flatten(), label='Targets')
plt.xlabel('Time')
plt.ylabel('Pitch')
plt.legend()
plt.title('Predictions vs. Targets')
plt.savefig('predictions_vs_targets.png')  # Save the plot as an image file


torch.save(model.state_dict(), 'last_save_trainmodel.pth')

# Evaluate the model
predictions, targets, eval_loss = evaluate_model(model, eval_loader, criterion)
print(f'Evaluation Loss: {eval_loss:.4f}')

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - targets) ** 2))
print(f'RMSE: {rmse:.4f}')
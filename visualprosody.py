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
import random
import wave

# Set random seeds for reproducibility
def set_random_seed(seed=23607):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call the function to set the random seed
set_random_seed()

# Configure logging
logging_model_name = "FFNN_mean_layer"  # or any other model name

# Handler for INFO level logs and higher (INFO, WARNING, ERROR, CRITICAL)
info_handler = logging.FileHandler(f'{logging_model_name}_training_visualprosody.log')
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Handler for ERROR level logs and higher (ERROR, CRITICAL)
error_handler = logging.FileHandler(f'{logging_model_name}_error.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Capture INFO level and above logs
    handlers=[info_handler, error_handler, console_handler]
)
# Function to compute the total duration of a video file
def get_video_duration(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_file}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

# Function to compute the total duration of all video files
def compute_total_video_duration(video_files):
    total_duration = 0
    for video_file in video_files:
        duration = get_video_duration(video_file)
        total_duration += duration
    return total_duration

# Function to compute the duration of a wav file
def get_wav_duration(wav_path):
    with wave.open(wav_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration
# Function to compute the total duration of a list of wav files
def compute_total_wav_duration(wav_files):
    total_duration = 0.0
    for wav_path in wav_files:
        duration = get_wav_duration(wav_path)
        total_duration += duration
    total_hours = total_duration / 3600  # Convert to hours
    print(f'Total duration of WAV files: {total_duration:.2f} seconds ({total_hours:.2f} hours)')
    return total_duration, total_hours

class VideoPitchDataset(Dataset):
    def __init__(self, base_dir, target_type='pitch', num_frames_per_target=2, padding_size=10):
        self.base_dir = base_dir
        self.rawfile_dir = base_dir + "/rawfile"
        self.target_type = target_type
        self.video_files = []
        self.feature_files = {}
        self.skipped_videos = []
        self.wav_files = []
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.num_frames_per_target = num_frames_per_target
        self.padding_size = padding_size
        self._load_data()

    def _load_data(self):
        total_videos = 0
        skipped_videos = 0
        skipped_videos2 = 0
        for subdir in os.listdir(self.rawfile_dir):
            subdir_path = os.path.join(self.rawfile_dir, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith(".mkv"):
                        total_videos += 1
                        video_file = os.path.join(subdir_path, file_name)
                        base_name = file_name.replace(".mkv", "")
                        pitch_file = os.path.join(self.base_dir, "pitch", f"{subdir}-pitch-{base_name}.npy")
                        duration_file = os.path.join(self.base_dir, "duration", f"{subdir}-duration-{base_name}.npy")
                        feature_file = os.path.join(self.base_dir, "feature", f"{subdir}-feature-{base_name}.pt")
                        wav_file = video_file.replace('.mkv', '.wav')
                        
                        if os.path.exists(video_file) and os.path.exists(pitch_file) and os.path.exists(wav_file):
                            if self._check_time_gap(video_file, wav_file):
                                self.video_files.append(video_file)
                                self.wav_files.append(wav_file)
                                self.feature_files[video_file] = {
                                    "pitch": pitch_file,
                                    "duration": duration_file,
                                    "feature": feature_file
                                }
                            else:
                                skipped_videos += 1
                                self.skipped_videos.append(video_file)
                                #logging.info(f"Skipped video file due to time gap: {video_file}")
                        else:
                            skipped_videos2 +=1
                            skipped_videos += 1
                            self.skipped_videos.append(video_file)
                            #logging.info(f"Skipped video file due to missing files: {video_file}")

        logging.info(f"Total number of video files: {total_videos}")
        logging.info(f"Total number of skipped video files: {skipped_videos}")
        logging.info(f"Total number of skipped video files(missing): {skipped_videos2}")
        #if self.skipped_videos:
            #logging.info(f"Skipped videos due to time gap: {self.skipped_videos}")


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        pitch = np.load(self.feature_files[video_file]["pitch"])
        duration = np.load(self.feature_files[video_file]["duration"])
        feature_file = self.feature_files[video_file]["feature"]

        if os.path.exists(feature_file):
            features = torch.load(feature_file)
            #logging.info(f"Loaded precomputed features from {feature_file}")
        else:
            # Extract features if not already saved
            video_data = self.preprocess_video(video_file)
            features = self.extract_features(video_data)
            torch.save(features, feature_file)
            logging.info(f"Extracted and saved features to {feature_file}")

        # Assign features based on duration
        grouped_features = self.assign_features(features, duration)
        #logging.info(f"Grouped features into {len(grouped_features)} groups based on duration.")

        # Pad or truncate features to match number of pitch values
        num_pitch_values = len(pitch)
        grouped_features = self.pad_or_truncate_features(grouped_features, num_pitch_values, self.padding_size)

        return grouped_features, torch.tensor(pitch, dtype=torch.float32)

    def _check_time_gap(self, video_file, wav_file):
        # Get video duration
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_file}")
            return False
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = video_frame_count / video_fps
        cap.release()

        # Get wav duration
        with wave.open(wav_file, 'r') as wav:
            wav_frame_count = wav.getnframes()
            wav_fps = wav.getframerate()
            wav_duration = wav_frame_count / wav_fps

        # Check if the time gap is within ±1/24
        time_gap = abs(video_duration - wav_duration)
        #print("time_gap: ",time_gap)
        return time_gap <= 1 / 24
    
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
            #features = hidden_states.mean(dim=1)  # Average pooling [num_frames, hidden_size]
             # Exclude the cls token (first token)
            hidden_states_without_cls = hidden_states[:, 1:, :]  # [num_frames, num_patches, hidden_size]
            features = hidden_states_without_cls.mean(dim=1)  # Average pooling [num_frames, hidden_size]
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

    def pad_or_truncate_features(self, features, target_length, padding_size):
        if len(features) > target_length:
            return features[:target_length]
        elif len(features) < target_length:
            padding = torch.zeros((padding_size, features.shape[1]))
            return torch.cat([features, padding], dim=0)[:target_length]
        return features

class FFNNModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, non_linearity=F.relu):
        super(FFNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.non_linearity = non_linearity

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        x = x.view(batch_size * seq_len, feature_size)
        x = self.non_linearity(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.non_linearity(layer(x))
        x = self.fc_out(x)
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
            #logging.info(f"Batch {batch_idx} - outputs shape: {outputs.shape}, targets shape: {targets.shape}")

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
base_dir = '/Users/khs/Downloads/dissertation/reference_works/hyunsik/visualprosody/CMD-1s-50-full'
#base_dir ='/mnt/parscratch/users/acp23hk/FS2_cmd/preprocessed_data/CMD-1s-50'
# Experiment configurations
configurations = [
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.001, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.0001, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 0.0},
    # Add more configurations as needed
]

configurations = [
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 1e-5},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 5*1e-4},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 1e-4},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 1e-3},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 0},
]



configurations = [
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.relu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.leaky_relu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.elu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.selu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.tanh, 'num_layers': 2, 'weight_decay': 0.0},
    # Add more configurations as needed
]


configurations = [
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.selu, 'num_layers': 2, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.selu, 'num_layers': 3, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.selu, 'num_layers': 4, 'weight_decay': 0.0},
    {'learning_rate': 0.01, 'padding_size': 10, 'non_linearity': F.selu, 'num_layers': 5, 'weight_decay': 0.0},
    # Add more configurations as needed
]

results = []

for config in configurations:
    logging.info(f"Starting experiment with config: {config}")
    set_random_seed()
    # Initialize the dataset and dataloader with the given padding size
    dataset = VideoPitchDataset(base_dir, padding_size=config['padding_size'])

    total_video_duration = compute_total_video_duration(dataset.video_files)
    total_wav_duration, total_wav_hours = compute_total_wav_duration(dataset.wav_files)
    print(f'Total duration of all video files: {total_video_duration:.2f} seconds ({total_video_duration / 3600:.2f} hours)')
    print(f'Total duration of all WAV files: {total_wav_duration:.2f} seconds ({total_wav_hours:.2f} hours)')

    logging.info(f'Total duration of all video files: {total_video_duration:.2f} seconds ({total_video_duration / 3600:.2f} hours)')
    logging.info(f'Total duration of all WAV files: {total_wav_duration:.2f} seconds ({total_wav_hours:.2f} hours)')
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = FFNNModel(hidden_size=128, num_layers=config['num_layers'], non_linearity=config['non_linearity'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    train_losses = []
    eval_losses = []

    start_epoch = 0
    checkpoint_path = f'checkpoint_step_20_{logging_model_name}_{config["learning_rate"]}_{config["num_layers"]}_{config["non_linearity"]}.pth'
    model, optimizer, start_epoch, train_losses, eval_losses, step = load_checkpoint(checkpoint_path, model, optimizer)

    # Training loop with checkpoint saving every 500 steps
    num_epochs = 5
    checkpoint_interval = 1000
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
                }, f'checkpoint_step_{step}_{logging_model_name}_{config["learning_rate"]}_{config["num_layers"]}_{config["non_linearity"]}.pth')

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logging.info(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}] Loss: {epoch_loss:.4f}')

        model.eval()
        predictions, targets, eval_loss = evaluate_model_FFNN(model, eval_loader, criterion)
        #logging.info(f"Predictions: {predictions}")
        #logging.info(f"Targets: {targets}")
        eval_losses.append(eval_loss)
        logging.info(f'Evaluation Loss after epoch {epoch + 1}: {eval_loss:.4f}')

    # Calculate MSE
    mse = np.mean((predictions - targets) ** 2)
    logging.info(f'Configuration: {config}, MSE: {mse:.4f}')
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    logging.info(f'Configuration: {config}, RMSE: {rmse:.4f}')
    results.append((config, mse, rmse))

    # Function to plot individual data elements
    def plot_individual_elements(predictions, targets, base_filename):
        num_elements = predictions.shape[0]

        for i in range(num_elements):
            plt.figure(figsize=(10, 5))
            plt.plot(predictions[i].flatten(), label='Predictions')
            plt.plot(targets[i].flatten(), label='Targets')
            plt.xlabel('Time')
            plt.ylabel('Pitch')
            plt.legend()
            plt.title(f'Predictions vs. Targets (Data Element {i + 1})')
            plt.savefig(f'{base_filename}_element_{i + 1}.png')  # Save the plot as an image file
            plt.close()

    # Plot training and evaluation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Evaluation Loss - {config}')
    plt.savefig(f'training_evaluation_loss_{config}.png')  # Save the plot as an image file

    # Plot predictions vs. targets
    plt.figure(figsize=(10, 5))
    plt.plot(predictions.flatten(), label='Predictions')
    plt.plot(targets.flatten(), label='Targets')
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.legend()
    plt.title(f'Predictions vs. Targets - {config}')
    plt.savefig(f'predictions_vs_targets_{config}.png')  # Save the plot as an image file

    plt.figure(figsize=(10, 5))
    plt.plot(predictions.flatten(), label='Predictions')
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.legend()
    plt.title(f'Predictions - {config}')
    plt.savefig(f'predictions_{config}.png')  # Save the plot as an image file

    plt.figure(figsize=(10, 5))
    plt.plot(targets.flatten(), label='Targets')
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.legend()
    plt.title(f'Targets - {config}')
    plt.savefig(f'targets_{config}.png')  # Save the plot as an image file

    torch.save(model.state_dict(), f'last_save_trainmodel_{logging_model_name}_{config["learning_rate"]}_{config["num_layers"]}_{config["non_linearity"]}.pth')

    # Plot predictions vs. targets for each data element
    #plot_individual_elements(predictions, targets, f'predictions_vs_targets_elements_{logging_model_name}_{config["learning_rate"]}_{config["padding_size"]}_{config["num_layers"]}')

# Log final results
for config, mse, rmse in results:
    logging.info(f'Configuration: {config}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

# After the experiment loop, log final counts
logging.info(f"Final count of video files used: {len(dataset.video_files)}")
logging.info(f"Final count of skipped video files: {len(dataset.skipped_videos)}")
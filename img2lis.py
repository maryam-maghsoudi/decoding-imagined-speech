import numpy as np
import mne
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import mne
from mne_bids import BIDSPath
import librosa
import pandas as pd
import torchaudio
import sys
import pdb
import math
import matplotlib.pyplot as plt

model_name = "openai/whisper-large-v3"  # You can change the model size: tiny, small, medium, large
# Load the processor (handles audio input preprocessing and text tokenization)
whisper_processor = WhisperProcessor.from_pretrained(model_name)

# MEG Data Loading Function
def load_meg_data(bids_path):
    # bids_path = BIDSPath(
    #     subject=subject_id,
    #     session=session_id,
    #     task=task,
    #     root=root_path,
    #     datatype="meg"
    # )
    raw = mne.read_epochs(bids_path)
    picks = dict(meg=True, eeg=False, stim=False, eog=False, ecg=False, misc=False)
    raw = raw.pick_types(**picks)
    data = raw.get_data()  # Shape: (n_epochs, n_channels, n_times)
    if data.ndim == 3:  # Ensure data is 3D before reshaping
        data = data.squeeze(0)  # (n_channels, n_samples)
    info = raw.info  # Use the info object from the original epochs
    raw = mne.io.RawArray(data, info)
    return raw

class MEGDataset(Dataset):
    def __init__(self, root_path, subject_id, modality):
        print(f"Initializing MEGDataset with data_path={root_path} and subject_id={subject_id}")
        self.root_path = root_path
        self.subject_id = subject_id
        self.sessions = [f'ses-{i}' for i in range(10)]
        if modality == 'lis':
            self.tasks = ['poem1lis', 'poem2lis']
        elif modality == 'img':
            self.tasks = ['poem1img', 'poem2img']
        self.data = []
        self.prepare_data()

    def prepare_data(self):
        for session in self.sessions:
            session_id = session.split('-')[1]
            for task in self.tasks:
                # Load MEG data
                meg_path = os.path.join(
                    self.root_path,
                    f'sub-{self.subject_id}',
                    session,
                    'meg',
                    f'sub-{self.subject_id}_sess-{session_id}_task-{task}_meg-epo.fif'
                )
                print(meg_path)
                if not os.path.exists(meg_path):
                    print("path not exist!")
                    continue
                raw = load_meg_data(meg_path)
                print(raw)
                data = raw.get_data()
                data_tensor = torch.from_numpy(data).float()

                # Append to dataset
                self.data.append({
                    'meg': data_tensor
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        meg = sample['meg']
        return meg

def chop_into_segments(data, segment_length, channel):
    channel_data = data[channel, :]
    num_segments = channel_data.shape[0] // segment_length
    truncated_data = channel_data[:num_segments * segment_length]
    segments = truncated_data.view(num_segments, segment_length)
    return segments

def chop_all_channels(data, segment_length):
    for ch in range(data.shape[0]):
         segments = chop_into_segments(data, segment_length, ch)
         if ch == 0:
            segments_all = np.zeros((data.shape[0], segments.shape[0], segments.shape[1]))
         segments_all[ch, :, :] = segments
    return segments_all


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]
# Sinusoidal positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)
# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, src, tgt):
        # Transformer expects shape (sequence_length, batch_size, embedding_dim)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer(src, tgt)
        return output.permute(1, 0, 2)  # Return to (batch_size, sequence_length, embedding_dim)


# Hyperparameters
#K = 1  # Total number of samples
input_dim = 256  # Dimension of each segment
N = np.floor(27001/ input_dim)   # Number of segments per sample
embedding_dim = 128  # Dimension of transformer input/output (d_model)
batch_size = 4
num_epochs = 30
learning_rate = 1e-4
validation_split = 0.2

root_path = '/fs/nexus-projects/brain_project/maryam_meg_dataset/BIDS'
print("Before MEGDataset initialization")
meg_dataset_lis = MEGDataset(root_path, '04', 'lis')
meg_dataset_img = MEGDataset(root_path, '04', 'img')

sess = 0
channel = 70
brain_img = meg_dataset_img[sess]
brain_lis = meg_dataset_lis[sess]
brain_img = chop_all_channels(brain_img, input_dim)
brain_lis = chop_all_channels(brain_lis, input_dim)
# brain_img = chop_into_segments(brain_img, input_dim, channel)
# brain_lis = chop_into_segments(brain_lis, input_dim, channel)

data_input = brain_img #(K, N, input_dim)
data_output = brain_lis
# Normalizing each sample
data_input = torch.tensor(brain_img, dtype=torch.float32)  # Convert to tensor
data_output = torch.tensor(brain_lis, dtype=torch.float32)  # Convert to tensor
data_input = F.normalize(data_input, p=2, dim=0)
data_output = F.normalize(data_output, p=2, dim=0)

num_validation = int(data_input.shape[0] * validation_split)
train_input = data_input[:-num_validation]
train_output = data_output[:-num_validation]
val_input = data_input[-num_validation:]
val_output = data_output[-num_validation:]

train_dataset = CustomDataset(train_input, train_output)
val_dataset = CustomDataset(val_input, val_output)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the linear layers and transformer model
input_linear = nn.Linear(input_dim, embedding_dim)
output_linear = nn.Linear(embedding_dim, input_dim)
transformer = TransformerModel(
    d_model=embedding_dim,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    dropout=0.1
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(
    list(input_linear.parameters()) +
    list(output_linear.parameters()) +
    list(transformer.parameters()),
    lr=learning_rate
)

train_losses = []
val_losses = []
# Training loop
for epoch in range(num_epochs):
    transformer.train()
    total_loss = 0
    for batch in train_dataloader:
        input_batch, target_batch = batch
        input_batch = input_batch.float()  # Ensure the batch is of type float
        target_batch = target_batch.float()

        # Forward pass
        optimizer.zero_grad()
        src = input_linear(input_batch)  # Shape: [batch_size, N, 128]
        tgt = input_linear(target_batch)  # Shape: [batch_size, N, 128]
        transformer_output = transformer(src, tgt)  # Shape: [batch_size, N, 128]
        output = output_linear(transformer_output)  # Shape: [batch_size, N, 256]

        # Compute loss
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation loop
    transformer.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_batch, target_batch = batch
            input_batch = input_batch.float()
            target_batch = target_batch.float()

            src = input_linear(input_batch)  # Shape: [batch_size, N, 128]
            tgt = input_linear(target_batch)  # Shape: [batch_size, N, 128]
            transformer_output = transformer(src, tgt)  # Shape: [batch_size, N, 128]
            output = output_linear(transformer_output)  # Shape: [batch_size, N, 256]

            loss = criterion(output, target_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
print("Training complete.")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig('loss_plot.png')  # Save the plot as a PNG file
print("Loss plot saved as 'loss_plot.png'.")


#-------------------------------------------------------
sample_idx = 0
input_sample, target_sample = val_dataset[sample_idx]
input_sample = input_sample.unsqueeze(0)  # Add batch dimension
target_sample = target_sample.unsqueeze(0)  # Add batch dimension

# Pass the input sample through the trained model
transformer.eval()
with torch.no_grad():
    src = input_linear(input_sample)  # Shape: [1, N, 128]
    tgt = input_linear(target_sample)  # Shape: [1, N, 128]
    transformer_output = transformer(src, tgt)  # Shape: [1, N, 128]
    output = output_linear(transformer_output)  # Shape: [1, N, 256]

# Convert output and target to numpy for plotting
output_np = output.squeeze(0).numpy()  # Remove batch dimension
target_np = target_sample.squeeze(0).numpy()

# Plot the comparison
plt.figure(figsize=(10, 5))
plt.plot(output_np[0, :], label='Model Output', linestyle='--')
plt.plot(target_np[0, :], label='Actual Target', linestyle='-')
plt.xlabel('Time Points')
plt.ylabel('Signal Amplitude')
plt.title('Model Output vs Actual Target')
plt.legend()
plt.grid()
plt.savefig('output_plot.png')  # Save the plot as a PNG file

cosine_similarity = 0
for iseg in range(output_np.shape[0]):
    dot_product = np.dot(output_np[100, :], target_np[100, :])
    norm_output = np.linalg.norm(output_np[100, :])
    norm_target = np.linalg.norm(target_np[100, :])
    cosine_similarity = cosine_similarity + dot_product / (norm_output * norm_target)
cosine_similarity = cosine_similarity/output_np.shape[0]
print(f"Cosine similarity of sample {sample_idx} {cosine_similarity}")

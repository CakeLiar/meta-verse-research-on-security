import os
import pandas as pd
import numpy as np

# Function to add random noise to a column
def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, len(data))
    return data + noise

# Walk through the dataset directory
base_dir = 'dataset_without_normalization_cleaned'
for movement in os.listdir(base_dir):
    movement_path = os.path.join(base_dir, movement)
    if os.path.isdir(movement_path):
        for participant in os.listdir(movement_path):
            participant_path = os.path.join(movement_path, participant)
            if os.path.isdir(participant_path):
                for session in os.listdir(participant_path):
                    session_path = os.path.join(participant_path, session)
                    if os.path.isdir(session_path):
                        for file in os.listdir(session_path):
                            if file.endswith('.csv'):
                                file_path = os.path.join(session_path, file)
                                
                                df = pd.read_csv(file_path)
                                
                                for col in df.columns:
                                    if any(x in col.lower() for x in ['pos', 'rot']):
                                        df[col] = add_noise(df[col])
                                
                                output_base = 'dataset_with_noise'
                                output_dir = os.path.join(output_base, movement, participant, session)
                                os.makedirs(output_dir, exist_ok=True)
                                
                                output_path = os.path.join(output_dir, file)
                                df.to_csv(output_path, index=False)

print("Noisy dataset saved in 'dataset_with_noise' folder.")
def add_normal_noise(data, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, len(data))
    return data + noise

def add_uniform_noise(data, noise_factor=0.01):
    noise = np.random.uniform(-noise_factor, noise_factor, len(data))
    return data + noise

def add_poisson_noise(data, noise_factor=0.01):
    noise = np.random.poisson(noise_factor, len(data))
    return data + noise

noise_functions = {
    'normal': add_normal_noise,
    'uniform': add_uniform_noise,
    'poisson': add_poisson_noise
}

base_dir = 'dataset_without_normalization_cleaned'
for movement in os.listdir(base_dir):
    movement_path = os.path.join(base_dir, movement)
    if os.path.isdir(movement_path):
        for participant in os.listdir(movement_path):
            participant_path = os.path.join(movement_path, participant)
            if os.path.isdir(participant_path):
                for session in os.listdir(participant_path):
                    session_path = os.path.join(participant_path, session)
                    if os.path.isdir(session_path):
                        for file in os.listdir(session_path):
                            if file.endswith('.csv'):
                                file_path = os.path.join(session_path, file)
                                
                                df = pd.read_csv(file_path)
                                
                                for noise_type, noise_func in noise_functions.items():
                                    noisy_df = df.copy()
                                    for col in noisy_df.columns:
                                        if any(x in col.lower() for x in ['pos', 'rot']):
                                            noisy_df[col] = noise_func(noisy_df[col])
                                    
                                    output_base = f'dataset_with_{noise_type}_noise'
                                    output_dir = os.path.join(output_base, movement, participant, session)
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    output_path = os.path.join(output_dir, file)
                                    noisy_df.to_csv(output_path, index=False)

print("Noisy datasets saved in respective folders.")


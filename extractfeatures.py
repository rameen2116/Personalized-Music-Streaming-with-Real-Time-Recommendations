import os
import csv
import librosa
import numpy as np
from tqdm import tqdm

# Function to extract MFCC features from audio file
def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.tolist()
    except Exception as e:
        print(f"Error extracting MFCC features from {audio_path}: {e}")
        return None

# Function to read tracks.csv and extract necessary columns
def read_tracks_csv(tracks_file):
    track_info = {}
    with open(tracks_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_id = row['track_id']
            genres_all = row['genres_all']
            title = row['title']
            track_info[track_id] = {'genres_all': genres_all, 'title': title}
    return track_info

# Function to process audio files in a directory and save MFCC features to CSV
def process_audio_directory(directory, tracks_info, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["track_id", "title", "genres_all", "mfcc_features"])
       
        total_files = sum(len(files) for _, _, files in os.walk(directory))
        for root, _, files in os.walk(directory):
            for file in tqdm(files, total=total_files, desc="Processing audio files"):
                if file.endswith(".mp3"):
                    audio_path = os.path.join(root, file)
                    track_id = os.path.splitext(file)[0]  # Extract track ID from file name
                    if track_id in tracks_info:
                        mfcc_features = extract_mfcc(audio_path)
                        if mfcc_features is not None:
                            writer.writerow([track_id, tracks_info[track_id]['title'], tracks_info[track_id]['genres_all'], mfcc_features])

# Main function
if __name__ == "__main__":
    # Specify the directory containing audio files
    fma_data_directory = "fma_large"
   
    # Specify the tracks CSV file
    tracks_csv = "tracks.csv"
   
    # Specify the output CSV file
    output_file = "mfcc_features_with_info.csv"
   
    # Read tracks CSV and extract necessary columns
    tracks_info = read_tracks_csv(tracks_csv)
   
    # Process audio files and save MFCC features with track info to CSV
    process_audio_directory(fma_data_directory, tracks_info, output_file)

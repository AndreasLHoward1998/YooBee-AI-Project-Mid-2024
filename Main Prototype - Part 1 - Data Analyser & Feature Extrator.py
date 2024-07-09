import pandas as pd
import numpy as np
import mne
from scipy.signal import butter, filtfilt

# This is the function to load and rename columns
def load_and_rename_csv(file_path):
    df = pd.read_csv(file_path, header=None, low_memory=False)  # Read without header
    # This will add the correct header/s
    new_columns = [
        'Sample Count',
        'EEG Channel Value: CP3',
        'EEG Channel Value: C3',
        'EEG Channel Value: F5',
        'EEG Channel Value: PO3',
        'EEG Channel Value: PO4',
        'EEG Channel Value: F6',
        'EEG Channel Value: C4',
        'EEG Channel Value: CP4',
        'Marker Column',
        'Timestamp'
    ]
    df.columns = new_columns

    # This will convert all columns to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# Below is the function to preprocess the EEG data (e.g., filtering)
def preprocess_eeg_data(df, sfreq=256):
    # This will convert timestamps from milliseconds to seconds, because the original form was too difficult to read
    df['Timestamp'] = df['Timestamp'] / 1000.0
    
    # Below will extract EEG channels
    eeg_channels = [
        'EEG Channel Value: CP3',
        'EEG Channel Value: C3',
        'EEG Channel Value: F5',
        'EEG Channel Value: PO3',
        'EEG Channel Value: PO4',
        'EEG Channel Value: F6',
        'EEG Channel Value: C4',
        'EEG Channel Value: CP4'
    ]
    
    eeg_data = df[eeg_channels].dropna().values.T

    # This applies the band-pass filter
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, axis=-1)
        return y
    
    filtered_data = bandpass_filter(eeg_data, 0.5, 50, sfreq)
    
    # This will create an MNE Raw object
    info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(filtered_data, info)
    
    return raw

# Here we have a function to extract features (e.g., power spectral density)
def extract_features(raw, sfreq=256):
    from mne.time_frequency import psd_array_multitaper

    # This will calculate power spectral density for each channel
    psds = []
    for data in raw.get_data():
        psd, freqs = psd_array_multitaper(data, sfreq, fmin=0.5, fmax=50, adaptive=True, normalization='full')
        psds.append(psd)

    psds = np.array(psds)
    
    # The code below will ensure the PSDs have the same length
    min_length = min(psd.shape[0] for psd in psds)
    psds = np.array([psd[:min_length] for psd in psds])
    freqs = freqs[:min_length]
    
    psd_df = pd.DataFrame(psds, index=raw.ch_names, columns=freqs)
    return psd_df

# This is the function to analyze drops in concentration, engagement, and memory commitment
def analyze_eeg_data(psd_df):
    # This will define frequency bands of interest
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)
    
    def band_power(psd, freqs, band):
        band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.mean(psd[:, band_idx], axis=1)
    
    theta_power = band_power(psd_df.values, psd_df.columns.astype(float), theta_band)
    alpha_power = band_power(psd_df.values, psd_df.columns.astype(float), alpha_band)
    beta_power = band_power(psd_df.values, psd_df.columns.astype(float), beta_band)
    
    engagement = beta_power / (theta_power + alpha_power)
    memory_commitment = alpha_power / (theta_power + beta_power)
    
    # This creates a DataFrame to hold the results
    analysis_df = pd.DataFrame({
        'Channel': psd_df.index,
        'Theta Power': theta_power,
        'Alpha Power': alpha_power,
        'Beta Power': beta_power,
        'Engagement': engagement,
        'Memory Commitment': memory_commitment
    })
    
    return analysis_df

# Here we have the main function to execute the workflow
def main(file_path):
    df = load_and_rename_csv(file_path)
    raw = preprocess_eeg_data(df)
    features = extract_features(raw)
    
    # This adds a "Time" header to the first cell
    features.index.name = 'Time'
    
    # This will analyze the EEG data for concentration, engagement, and memory commitment
    analysis = analyze_eeg_data(features)
    
    return features, analysis

# Below will execute the workflow with the provided file path
file_path = 'Prototype Dataset 1.csv'  # This will ensure this file is in the same directory as the script
features, analysis = main(file_path)

# The code below will save the features to a CSV file for analysis
features.to_csv('Extracted_Features.csv')

# Next, the code below will save the analysis to a CSV file for the AI to interpret
analysis.to_csv('EEG_Analysis.csv')

# Finally a print out of the sample of the analysis will be created so the user can double check
print(analysis.head())

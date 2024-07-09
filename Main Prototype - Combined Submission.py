import pandas as pd
import numpy as np
import mne
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import ttk
from docx import Document

# This is the function to load and rename columns in the csv file
def load_and_rename_csv(file_path):
    df = pd.read_csv(file_path, header=None, low_memory=False)  # This is to read without the header
    # For future work I can add the correct header here
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

    # This converts all columns to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# This is the function to preprocess EEG data (e.g., filtering)
def preprocess_eeg_data(df, sfreq=256):
    # This will convert timestamps to seconds, as it is normally in milliseconds and can cause confusion
    df['Timestamp'] = df['Timestamp'] / 1000.0
    
    # Extract the EEG channels from the csv
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
    
    # This creates an MNE Raw object to be used in the program
    info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(filtered_data, info)
    
    return raw

# This is the function to extract features from the interpreted data (e.g., power spectral density etc)
def extract_features(raw, sfreq=256):
    from mne.time_frequency import psd_array_multitaper

    # This will calculate power spectral density for each EEG channel
    psds = []
    for data in raw.get_data():
        psd, freqs = psd_array_multitaper(data, sfreq, fmin=0.5, fmax=50, adaptive=True, normalization='full')
        psds.append(psd)

    psds = np.array(psds)
    
    # This will ensure the PSDs all have the same length
    min_length = min(psd.shape[0] for psd in psds)
    psds = np.array([psd[:min_length] for psd in psds])
    freqs = freqs[:min_length]
    
    psd_df = pd.DataFrame(psds, index=raw.ch_names, columns=freqs)
    return psd_df

# This is the function to analyze drops in concentration, engagement, and memory commitment
def analyze_eeg_data(psd_df):
    # Below will define frequency bands of interest to help the AI
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
    
    # Creates a DataFrame to hold the results from the analysis
    analysis_df = pd.DataFrame({
        'Channel': psd_df.index,
        'Theta Power': theta_power,
        'Alpha Power': alpha_power,
        'Beta Power': beta_power,
        'Engagement': engagement,
        'Memory Commitment': memory_commitment
    })
    
    return analysis_df

# The main function to execute the workflow in the AI
def main(file_path):
    df = load_and_rename_csv(file_path)
    raw = preprocess_eeg_data(df)
    features = extract_features(raw)
    
    # This will add "Time" header to the first cell
    features.index.name = 'Time'
    
    # Below will analyze EEG data for concentration, engagement, and memory commitment
    analysis = analyze_eeg_data(features)
    
    return features, analysis

# This will execute the workflow with the provided file path
file_path = 'Prototype Dataset 1.csv'  # This will ensure this file is in the same directory as the script
features, analysis = main(file_path)

# Just to save features to a CSV file in case
features.to_csv('Extracted_Features.csv')

# Below will save analysis to a CSV file
analysis.to_csv('EEG_Analysis.csv')

# This will print out a sample of the analysis
print(analysis.head())

# This will load the CSV file
df = pd.read_csv('EEG_Analysis.csv')

# This will load the Word document I downloaded and put into the file
doc = Document('Intro to Machine Learning - activity.docx')

# This will extract text from the Word document
doc_text = '\n'.join([para.text for para in doc.paragraphs])

# This will print out the column names and first few rows for debugging
print("Column Names:", df.columns)
print(df.head())

# This will create a Tkinter window to display the educational material, but in it will be altered later
root = tk.Tk()
root.title("EEG Analysis and Document Display")

# This creates a Text widget to display the document text for the user
text_widget = tk.Text(root, wrap='word')
text_widget.pack(expand=True, fill='both')

# This will insert the document text into the Text widget so it can be seen
text_widget.insert('1.0', doc_text)

# This is the function to map values to colors
def value_to_color(value, value_range, color_range):
    """ Map a value to a color in a given range. """
    min_val, max_val = value_range
    min_color, max_color = color_range
    
    # Below will normalize the value to be within 0 and 1
    norm_value = (value - min_val) / (max_val - min_val)
    
    # Next we interpolate the color
    r = int(min_color[0] + (max_color[0] - min_color[0]) * norm_value)
    g = int(min_color[1] + (max_color[1] - min_color[1]) * norm_value)
    b = int(min_color[2] + (max_color[2] - min_color[2]) * norm_value)
    
    return f'#{r:02x}{g:02x}{b:02x}'

# Below is the function to update text color based on engagement values
def update_text_colors():
    for i, row in df.iterrows():
        engagement = row['Engagement']
        memory_commitment = row['Memory Commitment']
        
        # This will define the color ranges
        engagement_color = value_to_color(engagement, (0, 1), ((255, 0, 0), (0, 0, 255))) # Red to Blue
        memory_commitment_color = value_to_color(memory_commitment, (0, 1), ((0, 255, 0), (255, 255, 0))) # Green to Yellow
        
        text_widget.tag_add(f'engagement_{i}', f'{i + 1}.0', f'{i + 1}.end')
        text_widget.tag_configure(f'engagement_{i}', foreground=engagement_color, background=memory_commitment_color)

update_text_colors()

root.mainloop()

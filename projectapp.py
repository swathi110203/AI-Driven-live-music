import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import wave
from tensorflow.keras.models import load_model
import librosa
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import resampy
# Custom CSS style for dark blue background
instrument_classes = ["Piano", "Guitar", "Drums", "Vocals"]
LE = LabelEncoder().fit(instrument_classes)
custom_css = """
    <style>
        body {
            background-color: #2c3e50; /* Dark blue background color */
            color: white; /* Text color */
        }
    </style>
"""

# Function to add custom CSS
def set_custom_style():
    st.markdown(custom_css, unsafe_allow_html=True)

# Function to record audio from microphone
def record_audio(duration):
    sample_rate = 44100
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    return audio, sample_rate

# Function to save audio to temporary WAV file
def save_audio_to_temp_file(audio, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        wav_filename = fp.name
        with wave.open(fp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
    return wav_filename
model_path = "C:/Users/Sumanth/Dropbox/My PC (LAPTOP-U40MKBK6)/Downloads/audio_classification (1).hdf5"
model = load_model(model_path)
model_guitar_path="C:/Users/Sumanth/Dropbox/My PC (LAPTOP-U40MKBK6)/Downloads/audio_predictor_guitar.hdf5"
model_vocal_path="C:/Users/Sumanth/Dropbox/My PC (LAPTOP-U40MKBK6)/Downloads/audio_predictor-vocal.hdf5"
model_keys_path="C:/Users/Sumanth/Dropbox/My PC (LAPTOP-U40MKBK6)/Downloads/audio_predictor_keys.hdf5"
model_drums_path="C:/Users/Sumanth/Dropbox/My PC (LAPTOP-U40MKBK6)/Downloads/audio_predictor_drums.hdf5"
model_guitar = load_model(model_guitar_path)
model_vocal = load_model(model_vocal_path)
model_keys = load_model(model_keys_path)
model_drums = load_model(model_drums_path)
# Define instrument classes (replace with your own classes)
instrument_classes = ["Piano", "Guitar", "Vocals", "Drums"]

def predict_instrument(audio_file):
    audio,sample_rate=librosa.load(audio_file,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features =np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    prediction_class=LE.inverse_transform(predicted_label)
    return prediction_class


   
def extract_frequency_features(audio_file, duration=5, sample_rate=44100):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Split into sub-files of specified duration
    sub_files = librosa.effects.split(y, top_db=20, frame_length=duration*sample_rate, hop_length=duration*sample_rate)
    
    features = []
    
    for start, end in sub_files:
        sub_y = y[start:end]
        
        # Compute STFT
        D = librosa.stft(sub_y)
        
        # Compute magnitudes
        mag = np.abs(D)
        
        # Calculate average magnitude across time frames
        avg_mag = np.mean(mag, axis=1)
        
        # Calculate dB values
        db_values = librosa.amplitude_to_db(avg_mag)
        
        # Categorize dB values into frequency bands
        freq_bins = librosa.fft_frequencies(sr=sr)
        freq_categories = [calculate_frequency_category(freq) for freq in freq_bins]
        
        # Create dictionary to store category dB values, excluding '20000-inf'
        category_dbs = {category: [] for category in set(freq_categories) if category != '20000-inf'}
        
        for i in range(len(freq_bins)):
            category = freq_categories[i]
            if category != '20000-inf':
                category_dbs[category].append(db_values[i])
        
        # Take average of dB values for each category
        category_avg_dbs = {category: np.mean(values) for category, values in category_dbs.items()}
        
        # Sort category dB values based on frequency range
        sorted_category_avg_dbs = {k: category_avg_dbs[k] for k in sorted(category_avg_dbs.keys(), key=lambda x: float(x.split('-')[0]))}
        
        # Append to features list
        features.append(sorted_category_avg_dbs)
    
    return features


def calculate_frequency_category(freq):
    # Define frequency category ranges
    categories = [
        (20, 40), (40, 80), (80, 160), (160, 300), (300, 600),
        (600, 1200), (1200, 2400), (2400, 5000), (5000, 10000),
        (10000, 20000), (20000, np.inf)
    ]
    
    # Determine frequency category based on input frequency
    for i, (low, high) in enumerate(categories):
        if low <= freq < high:
            return f'{low}-{high}'  # Return category range as string
    return '20000-inf'  # For frequencies > 20 kHz
def main():
    set_custom_style() # Apply custom style

    st.title("Mix.AI")
    
    # Button to start listening
    listen_button = st.button("Listen")

    if listen_button:
        st.write("Listening...")
        audio_duration = 20  # Duration for recording audio in seconds
        audio, sample_rate = record_audio(audio_duration)
        st.write("Recording completed!")
        st.write("Playing back recorded audio...")
        wav_filename = save_audio_to_temp_file(audio, sample_rate)
        st.audio(wav_filename, format="audio/wav")
        detected_instrument = predict_instrument(wav_filename)
        st.write(f"Detected Instrument/Vocals: {detected_instrument}")
        audio_file_path = wav_filename
        features = extract_frequency_features(audio_file_path)
        df=pd.DataFrame(features)
        X = df[['20-40', '40-80', '80-160','160-300','300-600','600-1200','1200-2400','2400-5000','5000-10000','10000-20000']].values
        st.write(f'Current values:{X}')
        if detected_instrument==['Guitar']:
            st.write(model_guitar.predict(X))
        elif detected_instrument==['Vocals']:
            st.write(model_vocal.predict(X))
        elif detected_instrument==['Piano']:
            st.write(model_keys.predict(X))
        else:
            st.write(model_drums.predict(X))

# Run the app
if __name__ == "__main__":
    main()

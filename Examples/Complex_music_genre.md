# Real-Time Music Genre Classification System:

```matlab
% Step 1: Data Collection and Preprocessing

% Load an example music track
[x, Fs] = audioread('path_to_music_track.wav');  % Provide the actual path

% Display the original audio waveform
figure;
subplot(2,1,1);
plot(x);
title('Original Audio Signal');
xlabel('Sample');
ylabel('Amplitude');

% Normalize the audio to a standard amplitude range
x = x / max(abs(x));

% Perform noise reduction (e.g., simple moving average)
window_size = 1000;
smoothed_signal = movmean(x, window_size);

% Display the smoothed audio waveform
subplot(2,1,2);
plot(smoothed_signal);
title('Smoothed Audio Signal (Noise Reduction)');
xlabel('Sample');
ylabel('Amplitude');

% Step 2: Feature Extraction

% Parameters for MFCC computation
num_mfcc_coeffs = 13;
frame_length = 0.025;  % 25 ms frame length
frame_overlap = 0.01;  % 10 ms frame overlap

% Compute MFCCs using voicebox toolbox
mfccs = melcepst(smoothed_signal, Fs, 'M', num_mfcc_coeffs, floor(3*log(Fs)), frame_length, frame_overlap);

% Display the MFCCs
figure;
imagesc(1:size(mfccs, 1), 1:size(mfccs, 2), mfccs);
title('MFCCs (Mel-frequency Cepstral Coefficients)');
xlabel('Frame');
ylabel('MFCC Coefficient');
colorbar;

% Step 3: Model Training

% Assume 'mfccs' is the extracted MFCC features
% Assume 'genre_labels' contains corresponding genre labels (e.g., 'rock', 'pop', etc.)

% Train a Random Forest classifier
rf_model = TreeBagger(50, mfccs, genre_labels, 'Method', 'classification');

% Display the trained Random Forest model details
disp('Trained Random Forest Model:');
disp(rf_model);

% Step 4: Real-Time Genre Classification

% Initialize audio recording
Fs = 44100; % Sampling frequency
recObj = audiorecorder(Fs, 16, 1); % 16 bits, mono

disp('Real-Time Genre Classification started.');

while true
    % Start recording for a short duration
    disp('Recording...');
    recordblocking(recObj, 5); % Record for 5 seconds
    
    % Get the recorded audio data
    audio_data = getaudiodata(recObj);
    
    % Preprocess the audio data (e.g., noise reduction, audio segmentation)
    smoothed_signal = movmean(audio_data, 1000);
    
    % Extract MFCC features
    mfccs = melcepst(smoothed_signal, Fs, 'M', 13, floor(3*log(Fs)), 0.025, 0.01);
    
    % Predict the genre using the trained Random Forest model
    predicted_genre = predict(rf_model, mfccs);
    
    % Display the detected genre
    disp(['Detected Genre: ', mode(predicted_genre)]);
end
```

Replace `'path_to_music_track.wav'` with the actual path to your music track. Ensure you have the necessary toolboxes (e.g., Audio Toolbox, Statistics and Machine Learning Toolbox) for audio processing and machine learning functions used in the code.

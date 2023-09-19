# Real-Time Speech Emotion Recognition System

**Objective:**
Develop a system that can recognize and classify emotions (e.g., happy, sad, angry) from real-time speech input.

**Approach:**
We'll use a combination of speech processing techniques, feature extraction, and machine learning to achieve this.

### Project Outline:

1. **Data Collection and Preprocessing:**
   - Collect a dataset of speech samples labeled with different emotions (e.g., RAVDESS dataset).
   - Preprocess the audio data (e.g., noise reduction, audio segmentation).

2. **Feature Extraction:**
   - Extract relevant features from the audio data (e.g., Mel-frequency cepstral coefficients - MFCCs).

3. **Model Training:**
   - Train a machine learning model (e.g., Support Vector Machine - SVM) using the extracted features and labeled data.

4. **Real-Time Emotion Recognition:**
   - Develop a MATLAB program that captures real-time audio input.
   - Process the audio to extract MFCC features.
   - Use the trained model to classify the emotion in real-time.
   - Display the detected emotion to the user.

### Step 1: Data Collection and Preprocessing

**Explanation:**
- This step involves collecting a dataset of labeled speech samples and preprocessing the audio data to make it suitable for feature extraction.

### Step 2: Feature Extraction

**Explanation:**
- In this step, we extract Mel-frequency cepstral coefficients (MFCCs) from the preprocessed audio data. MFCCs are widely used in speech processing for feature representation.

### Step 3: Model Training

**Explanation:**
- In this step, we train a machine learning model, such as a Support Vector Machine (SVM), using the extracted MFCC features and corresponding emotion labels.

### Step 4: Real-Time Emotion Recognition

**Explanation:**
- This step involves creating a MATLAB program that continuously captures real-time audio input, processes it to extract MFCC features, and uses the trained model to classify the emotion in real-time. The detected emotion is then displayed to the user.

Great! Let's continue with Step 1: Data Collection and Preprocessing.

### Step 1: Data Collection and Preprocessing

#### Data Collection:
- **Objective:** Collect a dataset of labeled speech samples, where each sample is associated with a specific emotion (e.g., happy, sad, angry).
- **Explanation:** For this example, let's assume you already have a dataset like the RAVDESS dataset, which provides speech samples labeled with various emotions.

#### Data Preprocessing:
- **Objective:** Preprocess the audio data to make it suitable for feature extraction.
- **Explanation:** This typically involves noise reduction, audio segmentation (if needed), and ensuring a consistent format for input to the feature extraction step.


### MATLAB Code for Audio Preprocessing:

```matlab
% Load an example audio file
[x, Fs] = audioread('path_to_audio_file.wav');  % Provide the actual path

% Display the original audio waveform
figure;
subplot(2,1,1);
plot(x);
title('Original Audio Signal');
xlabel('Sample');
ylabel('Amplitude');

% Noise reduction (e.g., simple moving average)
window_size = 100;
smoothed_signal = movmean(x, window_size);

% Display the smoothed audio waveform
subplot(2,1,2);
plot(smoothed_signal);
title('Smoothed Audio Signal (Noise Reduction)');
xlabel('Sample');
ylabel('Amplitude');
```

**Explanation:**
- We start by loading an example audio file using `audioread`.
- We display the original audio waveform using `plot`.
- We apply noise reduction using a simple moving average.
- We display the smoothed audio waveform after noise reduction.

Certainly! Let's proceed to Step 2: Feature Extraction.

### Step 2: Feature Extraction

#### Feature Extraction:
- **Objective:** Extract relevant features from the preprocessed audio data.
- **Explanation:** We'll extract Mel-frequency cepstral coefficients (MFCCs) as they are commonly used for speech processing.

### MATLAB Code for MFCC Feature Extraction:

```matlab
% Assume 'smoothed_signal' is the preprocessed audio data

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
```

**Explanation:**
- We start by specifying the parameters for MFCC computation, such as the number of MFCC coefficients, frame length, and frame overlap.
- We use the `melcepst` function from the voicebox toolbox to compute MFCCs from the preprocessed audio data.
- We display the computed MFCCs.


### Step 3: Model Training

#### Model Training:
- **Objective:** Train a machine learning model using the extracted MFCC features and labeled data.
- **Explanation:** We'll use a supervised learning approach and train a model (e.g., Support Vector Machine - SVM) to classify the emotions based on the MFCC features.

Since this is a complex step, I'll provide a basic example using SVM for demonstration.

### MATLAB Code for SVM Model Training:

```matlab
% Assume 'mfccs' is the extracted MFCC features
% Assume 'emotion_labels' contains corresponding emotion labels (e.g., 1 for happy, 2 for sad, etc.)

% Train a Support Vector Machine (SVM) classifier
svm_model = fitcecoc(mfccs, emotion_labels);

% Display the trained SVM model details
disp('Trained SVM Model:');
disp(svm_model);
```

**Explanation:**
- We use the `fitcecoc` function to train a Support Vector Machine (SVM) classifier using the MFCC features and their corresponding emotion labels.

### Step 4: Real-Time Emotion Recognition

#### Real-Time Emotion Recognition:
- **Objective:** Create a MATLAB program that continuously captures real-time audio input, processes it to extract MFCC features, and uses the trained model to classify the emotion in real-time. The detected emotion is then displayed to the user.
- **Explanation:** This step involves integrating audio capture, MFCC feature extraction, and model prediction in real-time.

Let's write MATLAB code to demonstrate the real-time emotion recognition process.

### MATLAB Code for Real-Time Emotion Recognition:

```matlab
% Initialize audio recording
Fs = 44100; % Sampling frequency
recObj = audiorecorder(Fs, 16, 1); % 16 bits, mono

disp('Real-Time Emotion Recognition started.');

while true
    % Start recording for a short duration
    disp('Recording...');
    recordblocking(recObj, 3); % Record for 3 seconds
    
    % Get the recorded audio data
    audio_data = getaudiodata(recObj);
    
    % Preprocess the audio data (e.g., noise reduction, audio segmentation)
    smoothed_signal = movmean(audio_data, 100);
    
    % Extract MFCC features
    mfccs = melcepst(smoothed_signal, Fs, 'M', 13, floor(3*log(Fs)), 0.025, 0.01);
    
    % Predict the emotion using the trained SVM model
    predicted_emotion = predict(svm_model, mfccs);
    
    % Display the detected emotion
    disp(['Detected Emotion: ', predicted_emotion{1}]);
end
```

**Explanation:**
- We continuously capture audio for a short duration (here, 3 seconds).
- We preprocess the audio data by applying noise reduction (simple moving average) and then extract MFCC features.
- We use the trained SVM model to predict the emotion based on the extracted MFCC features.
- The detected emotion is displayed to the user.


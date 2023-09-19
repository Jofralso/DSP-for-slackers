Absolutely, let's provide MATLAB code examples and solutions for each topic, ranging from simple to difficult. We'll start with Topic 1: Introduction to Digital Signal Processing.

## Topic 1: Introduction to Digital Signal Processing

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Generate a sine wave signal and plot it

% Parameters
amplitude = 1;
frequency = 10; % 10 Hz
duration = 1;   % 1 second
sampling_rate = 1000; % 1000 samples per second

% Time vector
t = 0:1/sampling_rate:duration;

% Generate the sine wave signal
x = amplitude * sin(2*pi*frequency*t);

% Plot the signal
figure;
plot(t, x);
title('Sine Wave Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
```

**Explanation:**
- This simple example generates a sine wave signal with a frequency of 10 Hz and plots it.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Apply basic noise reduction to a signal

% Generate a noisy signal
t = linspace(0, 1, 1000);
noisy_signal = sin(2*pi*50*t) + 0.5*randn(size(t));

% Apply moving average for noise reduction
window_size = 10;
smoothed_signal = movmean(noisy_signal, window_size);

% Plot the original and smoothed signals
figure;
plot(t, noisy_signal, 'b', t, smoothed_signal, 'r', 'LineWidth', 1.5);
legend('Noisy Signal', 'Smoothed Signal');
title('Noise Reduction using Moving Average');
xlabel('Time (seconds)');
ylabel('Amplitude');
```

**Explanation:**
- This example generates a noisy signal, then applies a moving average for noise reduction.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Implement a digital filter using difference equation

% Define filter coefficients for a simple low-pass filter
b = [0.1 0.2 0.3 0.2 0.1];  % Numerator coefficients
a = 1;  % Denominator coefficients (for FIR filter, 'a' is 1)

% Generate a test input signal
x = randn(1, 100);  % Random input signal

% Apply the filter
filtered_signal = filter(b, a, x);

% Plot the original and filtered signals
figure;
plot(x, 'b', 'LineWidth', 1.5);
hold on;
plot(filtered_signal, 'r', 'LineWidth', 1.5);
legend('Original Signal', 'Filtered Signal');
title('Filtering using a Digital Filter');
xlabel('Sample');
ylabel('Amplitude');
```

**Explanation:**
- This example implements a simple low-pass digital filter using a difference equation and applies it to a random input signal.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Spectrum analysis using Fourier Transform

% Generate a signal with two sinusoidal components
Fs = 1000;  % Sampling frequency
t = 0:1/Fs:1;  % Time vector
x = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

% Plot the signal
figure;
plot(t, x);
title('Signal with Two Sinusoidal Components');
xlabel('Time (s)');
ylabel('Amplitude');

% Compute the Fourier Transform
X = fft(x);

% Compute the corresponding frequencies
frequencies = linspace(0, Fs, length(X));

% Plot the magnitude spectrum
figure;
plot(frequencies, abs(X));
title('Spectrum Analysis using Fourier Transform');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
```

**Explanation:**
- This example generates a signal with two sinusoidal components and performs spectrum analysis using the Fourier Transform.

## Topic 1: Introduction to Digital Signal Processing (Continued)

### MATLAB Code Example 5 (Difficult):

```matlab
% Difficult Example: Analyze a speech signal using Short-Time Fourier Transform (STFT)

% Load an example speech signal
[x, Fs] = audioread('speech.wav');
x = x(:, 1);  % Take only one channel if it's a stereo recording

% Parameters for STFT
window_length = 256;
overlap = 128;

% Compute STFT
[S, F, T] = spectrogram(x, hamming(window_length), overlap, window_length, Fs);

% Plot the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S)));
axis xy;
colorbar;
title('Spectrogram of Speech Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
```

**Explanation:**
- This difficult example loads a speech signal and analyzes it using the Short-Time Fourier Transform (STFT) to visualize its spectrogram.

### MATLAB Code Example 6 (Difficult):

```matlab
% Difficult Example: Implement a Kalman filter for tracking a moving object

% Simulate a 1D motion
t = 0:0.1:10;  % Time vector
true_position = sin(t);  % True position of the object
noisy_position = true_position + 0.2*randn(size(t));  % Noisy measurements

% Initialize Kalman filter parameters
initial_estimate = 1;
estimate = zeros(size(t));
estimate_error = zeros(size(t));

% Kalman filter loop
for i = 1:length(t)
    prediction = estimate(i);  % Predict next estimate
    prediction_error = estimate_error(i) + initial_estimate;  % Predict next estimate error
    
    % Update based on measurement
    kalman_gain = prediction_error / (prediction_error + 0.2);
    estimate(i) = prediction + kalman_gain * (noisy_position(i) - prediction);
    estimate_error(i) = (1 - kalman_gain) * prediction_error;
end

% Plot the true position and estimated position
figure;
plot(t, true_position, 'b', t, noisy_position, 'r.', t, estimate, 'g', 'LineWidth', 1.5);
legend('True Position', 'Noisy Measurements', 'Estimated Position (Kalman Filter)');
title('Object Tracking using Kalman Filter');
xlabel('Time');
ylabel('Position');
```

**Explanation:**
- This difficult example implements a Kalman filter to track the position of a moving object using noisy measurements.

## Topic 2: Discrete-Time Signals and Systems (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Generate a unit step function

n = -10:10;  % Discrete time values
u = @(n) n >= 0;  % Unit step function

% Plot the unit step function
stem(n, u(n));
title('Unit Step Function');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This simple example generates a unit step function using a defined function and plots it.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Generate a discrete impulse function

n = -10:10;  % Discrete time values
delta = @(n) n == 0;  % Discrete impulse function

% Plot the impulse function
stem(n, delta(n));
title('Discrete Impulse Function');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This simple example generates a discrete impulse function using a defined function and plots it.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Generate a discrete sinusoidal signal

n = 0:99;  % Discrete time values
A = 1;     % Amplitude
f = 0.04;  % Frequency

x = A * cos(2*pi*f*n);

% Plot the sinusoidal signal
stem(n, x);
title('Discrete Sinusoidal Signal');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This medium-hard example generates a discrete sinusoidal signal and plots it.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Implement convolution of two sequences

% Define the sequences
x = [1 2 3 4];  % Input sequence
h = [0.1 0.2 0.3];  % Impulse response

% Compute convolution
y = conv(x, h);

% Plot the original sequence, impulse response, and convolution result
figure;
subplot(3,1,1);
stem(x, 'b', 'LineWidth', 1.5);
title('Input Sequence');
xlabel('n');
ylabel('Amplitude');

subplot(3,1,2);
stem(h, 'r', 'LineWidth', 1.5);
title('Impulse Response');
xlabel('n');
ylabel('Amplitude');

subplot(3,1,3);
stem(y, 'g', 'LineWidth', 1.5);
title('Convolution Result');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This medium-hard example implements convolution of two sequences and plots the input sequence, impulse response, and the convolution result.

## Topic 3: Time-Domain and Frequency-Domain Analysis (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Compute and plot the autocorrelation of a signal

% Generate a simple signal
n = 0:100;  % Discrete time values
x = sin(0.2*pi*n) + 0.2*randn(size(n));  % Signal with noise

% Compute autocorrelation
autocorr_result = xcorr(x);

% Plot the autocorrelation
stem(autocorr_result);
title('Autocorrelation of a Signal');
xlabel('Lag');
ylabel('Autocorrelation');
```

**Explanation:**
- This simple example computes and plots the autocorrelation of a signal.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Compute and plot the cross-correlation of two signals

% Generate two simple signals
n = 0:100;  % Discrete time values
x = sin(0.1*pi*n);  % Signal 1
y = cos(0.1*pi*n);  % Signal 2

% Compute cross-correlation
crosscorr_result = xcorr(x, y);

% Plot the cross-correlation
stem(crosscorr_result);
title('Cross-correlation of Two Signals');
xlabel('Lag');
ylabel('Cross-correlation');
```

**Explanation:**
- This simple example computes and plots the cross-correlation of two signals.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Filtering a signal using convolution

% Generate a test signal
n = 0:999;  % Discrete time values
x = sin(0.02*pi*n) + 0.5*sin(0.2*pi*n);  % Signal with two components

% Design a simple low-pass filter
b = fir1(50, 0.1);  % FIR filter coefficients

% Apply the filter using convolution
filtered_signal = conv(x, b, 'same');

% Plot the original and filtered signals
figure;
plot(n, x, 'b', n, filtered_signal, 'r', 'LineWidth', 1.5);
legend('Original Signal', 'Filtered Signal');
title('Filtering a Signal using Convolution');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This medium-hard example filters a signal using convolution with a designed low-pass filter.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Implementing a moving average filter

% Generate a test signal
n = 0:999;  % Discrete time values
x = sin(0.02*pi*n) + 0.5*sin(0.2*pi*n);  % Signal with two components

% Design a moving average filter
window_size = 10;
b = ones(1, window_size) / window_size;

% Apply the filter using convolution
filtered_signal = conv(x, b, 'same');

% Plot the original and filtered signals
figure;
plot(n, x, 'b', n, filtered_signal, 'r', 'LineWidth', 1.5);
legend('Original Signal', 'Filtered Signal');
title('Moving Average Filtering of a Signal');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This medium-hard example implements a moving average filter to filter a signal.

## Topic 4: Z-Transform and Transfer Functions (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Compute and plot the Z-transform of a discrete signal

% Define a discrete signal (e.g., a step function)
n = 0:10;  % Discrete time values
x = ones(size(n));  % Unit step function

% Compute the Z-transform
[z, p, k] = tf2zpk(1, [1 -1], 1);

% Plot the Z-transform
figure;
zplane(z, p);
title('Z-transform of a Discrete Signal');
```

**Explanation:**
- This simple example computes and plots the Z-transform of a discrete signal.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Compute and plot the inverse Z-transform

% Define a simple Z-transform (e.g., 1 / (1 - 0.5*z^(-1)))
b = 1;  % Numerator coefficients
a = [1 -0.5];  % Denominator coefficients

% Compute the inverse Z-transform
[h, t] = impz(b, a, 10);

% Plot the inverse Z-transform
stem(t, h);
title('Inverse Z-transform of a Z-transform');
xlabel('n');
ylabel('Amplitude');
```

**Explanation:**
- This simple example computes and plots the inverse Z-transform of a Z-transform.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Compute and plot the frequency response of a digital filter

% Design a simple digital filter
b = [0.1 0.2 0.3 0.2 0.1];  % Numerator coefficients
a = 1;  % Denominator coefficients

% Compute the frequency response
[H, w] = freqz(b, a, 'whole');

% Plot the magnitude and phase of the frequency response
figure;
subplot(2,1,1);
plot(w, abs(H));
title('Frequency Response of a Digital Filter (Magnitude)');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');

subplot(2,1,2);
plot(w, angle(H));
title('Frequency Response of a Digital Filter (Phase)');
xlabel('Frequency (rad/sample)');
ylabel('Phase (radians)');
```

**Explanation:**
- This medium-hard example computes and plots the frequency response of a digital filter.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Design an IIR filter using the bilinear transformation

% Design a low-pass analog filter
[b, a] = butter(2, 0.2, 'low');  % 2nd-order Butterworth low-pass filter

% Perform bilinear transformation to obtain the discrete-time filter
[B, A] = bilinear(b, a, 1);

% Plot the frequency response of the IIR filter
fvtool(B, A);
title('Frequency Response of a Bilinear-transformed IIR Filter');
```

**Explanation:**
- This medium-hard example designs an IIR filter using the bilinear transformation and plots its frequency response.

## Topic 5: Filter Design (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Design a simple FIR low-pass filter

% Design the filter
N = 50; % Order of the filter
cutoff_freq = 0.1; % Cutoff frequency

b = fir1(N, cutoff_freq); % Design the filter using fir1

% Plot the magnitude response
freqz(b, 1);
title('Frequency Response of FIR Low-Pass Filter');
```

**Explanation:**
- This simple example designs a low-pass FIR filter and plots its frequency response.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Design a simple IIR low-pass filter using butterworth

% Design the filter
N = 4; % Order of the filter
cutoff_freq = 0.1; % Cutoff frequency

[b, a] = butter(N, cutoff_freq); % Design the filter using butter

% Plot the magnitude response
freqz(b, a);
title('Frequency Response of IIR Low-Pass Filter (Butterworth)');
```

**Explanation:**
- This simple example designs a low-pass IIR filter using Butterworth approximation and plots its frequency response.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Design a high-pass FIR filter using window method

% Design the filter
N = 100; % Order of the filter
cutoff_freq = 0.2; % Cutoff frequency

b = fir1(N, cutoff_freq, 'high'); % Design the filter using fir1

% Plot the magnitude response
freqz(b, 1);
title('Frequency Response of FIR High-Pass Filter');
```

**Explanation:**
- This medium-hard example designs a high-pass FIR filter using the window method and plots its frequency response.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Design a band-pass IIR filter using elliptic approximation

% Design the filter
N = 6; % Order of the filter
Wp = [0.1 0.5]; % Passband frequencies
Ws = [0.05 0.6]; % Stopband frequencies

[b, a] = ellip(N, 0.5, 40, Wp, 'bandpass');

% Plot the magnitude response
freqz(b, a);
title('Frequency Response of IIR Band-Pass Filter (Elliptic)');
```

**Explanation:**
- This medium-hard example designs a band-pass IIR filter using the elliptic approximation and plots its frequency response.

## Topic 6: Advanced Topics (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Compute the Discrete Fourier Transform (DFT) of a signal

% Generate a simple signal
n = 0:99;  % Discrete time values
x = sin(0.1*pi*n) + 0.5*sin(0.2*pi*n);  % Signal with two components

% Compute the DFT
X = fft(x);

% Plot the magnitude spectrum
figure;
stem(abs(X));
title('Magnitude Spectrum using Discrete Fourier Transform');
xlabel('Frequency (bins)');
ylabel('Magnitude');
```

**Explanation:**
- This simple example computes the Discrete Fourier Transform (DFT) of a signal and plots its magnitude spectrum.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Compute the inverse Discrete Fourier Transform (IDFT)

% Generate a simple spectrum
N = 100;
X = zeros(1, N);
X(10) = 10;
X(N-10+2) = 10;

% Compute the IDFT
x = ifft(X);

% Plot the original signal
figure;
stem(abs(x));
title('Reconstructed Signal using Inverse DFT');
xlabel('Sample');
ylabel('Amplitude');
```

**Explanation:**
- This simple example computes the inverse Discrete Fourier Transform (IDFT) to reconstruct a signal from its spectrum.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Implement the Fast Fourier Transform (FFT) algorithm

% Generate a simple signal
N = 128;  % Number of samples
x = sin(2*pi*0.05*(0:N-1)) + 0.5*sin(2*pi*0.2*(0:N-1));

% Compute FFT using built-in function
X_fft_builtin = fft(x);

% Compute FFT using implemented FFT function
X_fft_custom = my_fft(x);

% Check the difference between built-in and custom FFT
diff_fft = max(abs(X_fft_builtin - X_fft_custom));

disp(['Maximum difference between FFTs: ', num2str(diff_fft)]);

function X = my_fft(x)
    N = length(x);
    if N == 1
        X = x;
    else
        x_even = my_fft(x(1:2:N-1));
        x_odd = my_fft(x(2:2:N));
        factor = exp(-2i*pi/N).^(0:N/2-1);
        X = [x_even + factor .* x_odd, x_even - factor .* x_odd];
    end
end
```

**Explanation:**
- This medium-hard example implements the Fast Fourier Transform (FFT) algorithm and compares it with the built-in MATLAB FFT function.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Compute and plot the spectrogram of an audio signal

% Load an example audio signal
[x, Fs] = audioread('speech.wav');
x = x(:, 1);  % Take only one channel if it's a stereo recording

% Parameters for spectrogram
window_length = 512;
overlap = window_length / 2;
nfft = 1024;

% Compute spectrogram
[S, F, T] = spectrogram(x, hamming(window_length), overlap, nfft, Fs);

% Plot the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S)));
axis xy;
colorbar;
title('Spectrogram of Audio Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
```

**Explanation:**
- This medium-hard example computes and plots the spectrogram of an audio signal.

## Topic 7: Practical Applications (Continued)

### MATLAB Code Example 1 (Simple):

```matlab
% Simple Example: Apply noise reduction to an audio signal

% Load an example noisy audio signal
[x, Fs] = audioread('noisy_audio.wav');
x = x(:, 1);  % Take only one channel if it's a stereo recording

% Apply noise reduction (e.g., simple moving average)
window_size = 100;
smoothed_signal = movmean(x, window_size);

% Plot the original and smoothed signals
figure;
subplot(2,1,1);
plot(x);
title('Original Noisy Audio Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2,1,2);
plot(smoothed_signal);
title('Smoothed Audio Signal (Noise Reduction)');
xlabel('Sample');
ylabel('Amplitude');
```

**Explanation:**
- This simple example applies noise reduction to an audio signal using a simple moving average.

### MATLAB Code Example 2 (Simple):

```matlab
% Simple Example: Equalize an audio signal

% Load an example audio signal
[x, Fs] = audioread('audio.wav');
x = x(:, 1);  % Take only one channel if it's a stereo recording

% Design an equalization filter (e.g., boosting higher frequencies)
equalizer_filter = designParamEQ('Frequency', [500 5000], 'Gain', [0 10], 'Fs', Fs);

% Apply the equalization filter
equalized_signal = equalizer_filter(x);

% Plot the original and equalized signals
figure;
subplot(2,1,1);
plot(x);
title('Original Audio Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2,1,2);
plot(equalized_signal);
title('Equalized Audio Signal');
xlabel('Sample');
ylabel('Amplitude');
```

**Explanation:**
- This simple example equalizes an audio signal by boosting higher frequencies.

### MATLAB Code Example 3 (Medium-Hard):

```matlab
% Medium-Hard Example: Apply image filtering using a discrete filter

% Load and display an example image
image = imread('cameraman.tif');
imshow(image);
title('Original Image');

% Design a simple filter (e.g., edge detection)
filter = [-1 -1 -1; -1 8 -1; -1 -1 -1];

% Apply the filter using convolution
filtered_image = conv2(double(image), filter, 'same');

% Display the filtered image
figure;
imshow(uint8(filtered_image));
title('Filtered Image (Edge Detection)');
```

**Explanation:**
- This medium-hard example applies image filtering using a discrete filter for edge detection.

### MATLAB Code Example 4 (Medium-Hard):

```matlab
% Medium-Hard Example: Image rotation using Discrete Fourier Transform (DFT)

% Load and display an example image
image = imread('peppers.png');
imshow(image);
title('Original Image');

% Compute the Discrete Fourier Transform (DFT) of the image
dft_image = fft2(double(image));

% Rotate the image by 45 degrees
rotated_dft_image = imrotate(dft_image, 45, 'bicubic', 'crop');

% Compute the inverse DFT to obtain the rotated image
rotated_image = abs(ifft2(rotated_dft_image));

% Display the rotated image
figure;
imshow(uint8(rotated_image));
title('Rotated Image (45 degrees)');
```

**Explanation:**
- This medium-hard example rotates an image by 45 degrees using the Discrete Fourier Transform (DFT).

I'm glad you're finding this helpful! Let's continue exploring further practical applications.

### MATLAB Code Example 5 (Difficult):

```matlab
% Difficult Example: Implement a real-time audio spectrum analyzer

% Initialize audio recording
Fs = 44100; % Sampling frequency
duration = 10; % Duration of recording in seconds
recObj = audiorecorder(Fs, 16, 1); % 16 bits, mono

% Start recording
disp('Start speaking.');
recordblocking(recObj, duration);
disp('End of recording.');

% Get the recorded audio data
audio_data = getaudiodata(recObj);

% Compute the spectrogram
window_length = 1024;
overlap = 512;
nfft = 1024;
[S, F, T] = spectrogram(audio_data, hamming(window_length), overlap, nfft, Fs);

% Plot the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S)));
axis xy;
colorbar;
title('Real-Time Audio Spectrum Analyzer');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
```

**Explanation:**
- This difficult example implements a real-time audio spectrum analyzer using MATLAB, which displays the spectrogram of the recorded audio in real-time.

### MATLAB Code Example 6 (Difficult):

```matlab
% Difficult Example: Implement a real-time face detection using webcam

% Create a video input object
vidObj = videoinput('winvideo', 1);

% Set the video input parameters
set(vidObj, 'FramesPerTrigger', 1);
set(vidObj, 'TriggerRepeat', Inf);
triggerconfig(vidObj, 'manual');

% Start the video acquisition
start(vidObj);

% Create a figure for display
figure;

while ishandle(vidObj)
    % Trigger the video input to capture a frame
    trigger(vidObj);
    img = getdata(vidObj, 1);
    
    % Perform face detection
    bbox = step(faceDetector, img);
    if ~isempty(bbox)
        % Draw bounding boxes around detected faces
        img = insertObjectAnnotation(img, 'rectangle', bbox, 'Face');
    end
    
    % Display the image with detected faces
    imshow(img);
    title('Real-Time Face Detection');
    
    % Pause for a moment
    pause(0.1);
end

% Stop the video acquisition and clear the video input object
stop(vidObj);
delete(vidObj);
clear vidObj;
```

**Explanation:**
- This difficult example implements real-time face detection using the webcam and MATLAB's Computer Vision Toolbox.
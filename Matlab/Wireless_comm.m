% Simulation parameters
num_symbols = 1000;  % Number of QPSK symbols
symbol_rate = 1000;  % Symbol rate (symbols per second)

% Generate random QPSK symbols
QPSK_symbols = (2 * (randi([0, 1], 1, num_symbols) - 0.5)) + 1i * (2 * (randi([0, 1], 1, num_symbols) - 0.5));

% Modulate using QPSK
modulated_signal = sqrt(0.5) * QPSK_symbols;

% Create a multipath fading channel response
channel_delay = [0, 1, 3, 5];  % Delay taps in samples
channel_response = [1, 0.8, 0.4, 0.2];  % Channel gains

% Apply the channel response (convolution)
received_signal = conv(modulated_signal, channel_response, 'same');

% Add AWGN (noise)
snr_db = 10;  % Signal-to-noise ratio in dB
noise_variance = 0.5 / (10^(snr_db / 10));  % Noise variance
noise = sqrt(noise_variance) * (randn(1, num_symbols) + 1i * randn(1, num_symbols));
received_signal_with_noise = received_signal + noise;

% Demodulate (matched filter)
demodulated_signal = received_signal_with_noise;

% Plot signals in time and frequency domains
figure;

% Time domain plots
subplot(2, 1, 1);
plot(real(received_signal), 'b', 'LineWidth', 1.2, 'DisplayName', 'Received Signal (Real)');
hold on;
plot(imag(received_signal), 'r', 'LineWidth', 1.2, 'DisplayName', 'Received Signal (Imag)');
xlabel('Time');
ylabel('Amplitude');
title('Received Signal in Time Domain');
legend;

% Frequency domain plots
subplot(2, 1, 2);
plot_spectrum(received_signal, symbol_rate, 'Received Signal', 'b', 'LineWidth', 1.2);
hold on;
plot_spectrum(modulated_signal, symbol_rate, 'Modulated Signal', 'r', 'LineWidth', 1.2);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Spectrum of Signals');
legend;

sgtitle('Received Signals in Time and Frequency Domains');

function plot_spectrum(signal, fs, label, color, varargin)
    N = length(signal);
    f = (-N/2:N/2-1) * fs / N;
    spectrum = fftshift(fft(signal));
    magnitude = 10*log10(abs(spectrum));
    plot(f, magnitude, 'Color', color, varargin{:});
    xlim([-fs/2 fs/2]);
    legend(label);
end

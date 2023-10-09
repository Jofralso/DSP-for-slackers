% Simulation parameters
num_bits = 1000;  % Number of bits
bit_rate = 1000;  % Bit rate (bits per second)

% Generate random binary data
data = randi([0, 1], 1, num_bits);

% Modulate using BPSK
modulated_signal = 2*data - 1;

% Add AWGN (noise)
snr_db = 10;  % Signal-to-noise ratio in dB
noise_variance = 0.5 / (10^(snr_db / 10));  % Noise variance
noise = sqrt(noise_variance) * randn(1, num_bits);
received_signal = modulated_signal + noise;

% Demodulate
demodulated_signal = sign(received_signal);

% Calculate energy and power
energy_data = sum(data.^2);
energy_modulated = sum(modulated_signal.^2);
energy_received = sum(received_signal.^2);
energy_demodulated = sum(demodulated_signal.^2);

average_power_data = mean(data.^2);
average_power_modulated = mean(modulated_signal.^2);
average_power_received = mean(received_signal.^2);
average_power_demodulated = mean(demodulated_signal.^2);

disp('Energy and Power calculations:');
disp('---------------------------------');
disp('Energy of transmitted data (in bits):');
disp(energy_data);
disp('Energy of modulated signal:');
disp(energy_modulated);
disp('Energy of received signal:');
disp(energy_received);
disp('Energy of demodulated signal:');
disp(energy_demodulated);

disp('Average power of transmitted data (per bit):');
disp(average_power_data);
disp('Average power of modulated signal:');
disp(average_power_modulated);
disp('Average power of received signal:');
disp(average_power_received);
disp('Average power of demodulated signal:');
disp(average_power_demodulated);


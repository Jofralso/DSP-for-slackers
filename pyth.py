import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import sys

# Parameters for audio processing
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Create an audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the figure for displaying the pattern
plt.ion()
fig, ax = plt.subplots()
colormap = 'jet'
ax.set_title('Real-Time Reactive Psychedelic Art Generator')
psychedelic_plot = ax.imshow(np.zeros((500, 500)), cmap=colormap, vmin=-1, vmax=1)

# Initialize modulation parameters
modulation_amplitude = 1.0
modulation_frequency = 1.0

print('Real-time audio processing started. Press Ctrl+C to stop.')

try:
    while True:
        # Read audio data in real-time
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Normalize the audio data to the range [0, 1]
        normalized_audio = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))

        # Update modulation parameters based on audio amplitude
        modulation_amplitude = np.max(normalized_audio)
        modulation_frequency = 10 * modulation_amplitude  # Adjust the factor as needed

        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 500), np.linspace(-np.pi, np.pi, 500))

        # Generate the psychedelic pattern based on audio modulation
        psychedelic_pattern = np.sin(np.sqrt((x * modulation_frequency)**2 + (y * modulation_frequency)**2)) \
                              * np.cos(x + y)

        # Display the psychedelic pattern in real-time
        psychedelic_plot.set_array(psychedelic_pattern)
        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    pass

print('Real-time audio processing stopped.')

# Close the audio stream
stream.stop_stream()
stream.close()
p.terminate()

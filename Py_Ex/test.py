import numpy as np
import moviepy.editor as mp

# Create a function to generate frames based on the music
def make_frame(t):
    # Example: Modulate the size of a rectangle with the audio amplitude
    amplitude = np.sin(2 * np.pi * t)
    width = 200 + 100 * amplitude  # Adjust as needed
    height = 100 + 50 * amplitude   # Adjust as needed
    duration = 0.03  # Duration of each frame in seconds

    # Create a blank image with white background
    img = np.zeros((200, 300, 3), dtype=np.uint8) + 255

    # Draw a rectangle with varying size based on amplitude
    start_point = (50, 50)
    end_point = (50 + int(width), 50 + int(height))
    color = (0, 0, 255)  # Red color in BGR
    thickness = -1  # Fill the rectangle

    # Draw rectangle
    img[start_point[1]:end_point[1], start_point[0]:end_point[0]] = color

    return img

# Create an audio clip
audio = mp.AudioFileClip("C:/Users/z3rt/Documents/GitHub/DSP-for-those-who -didnt-get-it/DragonBallZ.mp3")

# Set the duration of the animation based on the audio duration
duration = audio.duration

# Create the animation
animation = mp.VideoClip(make_frame, duration=duration)

# Set the audio for the animation
animation = animation.set_audio(audio)

# Set the desired video parameters (e.g., resolution, framerate)
animation = animation.set_fps(30).set_duration(duration)

# Write the animation to a file
animation.write_videofile("reactive_animation.mp4", fps=30)

import numpy as np
import matplotlib.pyplot as plt

# Encode a message into an image using FFT
def encode_message(message, image_shape):
    # Pad the message to match the image size
    padded_message = message.ljust(np.prod(image_shape))
    # Reshape the message to match the image shape
    reshaped_message = np.array(list(padded_message)).reshape(image_shape)
    # Convert characters to ASCII values and use as the pixel values
    encoded_image = np.array([[ord(char) for char in row] for row in reshaped_message])
    return np.fft.fft2(encoded_image)

# Decode the message from the encoded image using inverse FFT
def decode_message(encoded_image):
    decoded_image = np.fft.ifft2(encoded_image)
    decoded_message = "".join([chr(int(round(val.real))) for val in decoded_image.flatten()])
    return decoded_message

# Define the message and image dimensions
message = "And unto Adam he said, Because thou hast hearkened unto the voice of thy wife, and hast eaten of the tree, of which I commanded thee, saying, Thou shalt not eat of it: cursed is the ground for thy sake; in sorrow shalt thou eat of it all the days of thy life; 3:18 Thorns also and thistles shall it bring forth to thee; and thou shalt eat the herb of the field; 3:19 In the sweat of thy face shalt thou eat bread, till thou return unto the ground; for out of it wast thou taken: for dust thou art, and unto dust shalt thou return."
image_shape = (25, 25)

# Encode the message into an image
encoded_image = encode_message(message, image_shape)

# Display the encoded message (FFT of the message)
plt.imshow(np.abs(encoded_image), cmap='gray')
plt.title('Encoded Message (FFT)')
plt.show()

# Decode the message from the encoded image
decoded_message = decode_message(encoded_image)
print('Decoded Message:', decoded_message)

import numpy as np
from PIL import Image

# 1. import image file
image_path = "dogNbike.jpg"
image = Image.open(image_path)
image_array = np.array(image)  # convert image to numpy

# 2. (Height, Width, Channel) -> (Batch, Height, Width, Channel)
expanded_image = np.expand_dims(image_array, axis=0)
print("Shape after expand_dims:", expanded_image.shape)

# 3. (Batch, Height, Width, Channel) -> (Batch, Channel, Width, Height)
transposed_image = np.transpose(expanded_image, (0, 3, 2, 1))
print("Shape after transpose:", transposed_image.shape)

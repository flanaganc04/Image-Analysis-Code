from PIL import Image, ImageOps

# Load the image
image_path = "circleBinary.png"
image = Image.open(image_path)

# Ensure the image is in binary mode (black and white)
binary_image = image.convert('1')

# Invert the image
inverted_image = ImageOps.invert(binary_image)

# Save the inverted image
inverted_image.save("circleBinaryInvert.png")

# Optionally, display the original and inverted images
# image.show()
inverted_image.show()
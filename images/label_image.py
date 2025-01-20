# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add sensor labels to picture of device
"""

__author__ = "Kyle Vitautas Lopin"

# installed librarise
from PIL import Image, ImageDraw, ImageFont

# Open the existing JPEG image
image = Image.open("device_image.jpg")
# crop the image
image = image.crop((150, 900, 2150, 2200))  # (left, upper, right, lower)

# Create a drawing object
draw = ImageDraw.Draw(image)

# Optional: Set a font (requires a .ttf file, or use default if not available)
font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 50)  # Adjust font size as needed
# font = ImageFont.load_default()

# Add text to the image
draw.text((240, 90), "AS7262", fill="black", font=font)
draw.text((650, 90), "AS7263", fill="black", font=font)
draw.text((1570, 45), "AS7265x", fill="black", font=font)


# Save the modified image as a new JPEG
image.save("device_final.jpeg", "JPEG")

image.show()

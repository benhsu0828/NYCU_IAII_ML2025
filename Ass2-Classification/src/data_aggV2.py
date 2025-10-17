import os
import random
from PIL import Image, UnidentifiedImageError

# Define paths
test_final_dir = "test-final"
backgrounds_dir = "backgrounds"
output_dir = "test-final-with-backgrounds"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get image lists
test_images = [f for f in os.listdir(test_final_dir) if f.endswith((".jpg", ".png"))]
background_images = [f for f in os.listdir(backgrounds_dir) if f.endswith((".jpg", ".png"))]

# Process each test image
for test_image_name in test_images:
    test_image_path = os.path.join(test_final_dir, test_image_name)
    try:
        # Open test image
        img = Image.open(test_image_path).convert("RGBA")

        # Create a mask from the black and white background
        mask = Image.new("L", img.size, 0)
        for x in range(img.width):
            for y in range(img.height):
                r, g, b, a = img.getpixel((x, y))
                if (r < 20 and g < 20 and b < 20) or (r > 235 and g > 235 and b > 235):
                    mask.putpixel((x, y), 0)
                else:
                    mask.putpixel((x, y), 255)

        # Choose a random background
        background_image_name = random.choice(background_images)
        background_image_path = os.path.join(backgrounds_dir, background_image_name)
        background_image = Image.open(background_image_path).convert("RGBA")

        # Add random padding
        padding_x = random.randint(10, 30)
        padding_y = random.randint(10, 30)
        new_size = (img.width + 2 * padding_x, img.height + 2 * padding_y)

        # Resize background to new size
        background_image = background_image.resize(new_size)

        # Paste test image onto the background
        paste_position = (padding_x, padding_y)
        background_image.paste(img, paste_position, mask)

        # Save the new image
        output_image_path = os.path.join(output_dir, test_image_name)
        background_image.save(output_image_path, "PNG")
    except UnidentifiedImageError:
        print(f"Cannot identify image file: {test_image_path}")
    except Exception as e:
        print(f"Error processing {test_image_path}: {e}")

print("Processing complete.")

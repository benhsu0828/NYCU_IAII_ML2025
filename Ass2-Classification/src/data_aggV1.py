import os
import torchvision.transforms.v2 as T
from PIL import Image
import torch

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    """
    Add speckle noise to the image.
    """
    def __init__(self, noise_level=0.1):
        """
        :param noise_level: Standard deviation of the noise distribution
        """
        self.noise_level = noise_level

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image on which noise is added
        :return: PyTorch tensor, image with speckle noise
        """
        # Generate speckle noise
        noise = torch.randn_like(tensor) * self.noise_level

        # Add speckle noise to the image
        noisy_tensor = tensor * (1 + noise)

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

class AddPoissonNoise(object):
    """
    Add Poisson noise to the image.
    """
    def __init__(self, lam=1.0):
        """
        :param lam: Lambda parameter for Poisson distribution
        """
        self.lam = lam

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image to which noise is added
        :return: PyTorch tensor, image with Poisson noise
        """
        # Generate Poisson noise
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))

        # Add Poisson noise to the image
        noisy_tensor = tensor + noise / 255.0  # Assuming the image is scaled between 0 and 1

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1  # Salt noise: setting some pixels to 1
        tensor[(noise > 1 - self.pepper_prob)] = 0  # Pepper noise: setting some pixels to 0
        return tensor

# Define the image augmentation transformations
transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to tensor

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.1),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
    T.ToPILImage()  # Convert tensor back to PIL image for saving

])

# Source folder containing the original images
src_folder = "test-in"

# Destination folder to save the augmented images
dst_folder = "test-out"

# Create the destination folder if it doesn't exist
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Loop through each file in the source folder
i=0
for filename in os.listdir(src_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image
        print("Reading ... ", filename)
        img_path = os.path.join(src_folder, filename)
        img = Image.open(img_path).convert("RGB")

        # Apply the transformations
        img_augmented = transform(img)

        # Save the augmented image
        print("Saving ... ", filename)
        save_path = os.path.join(dst_folder, filename)
        img_augmented.save(save_path, "JPEG")
        i=i+1
        print(i)

print("Image augmentation completed.")

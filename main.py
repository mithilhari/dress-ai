import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateImagesConfig

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

# TODO(developer): Update and un-comment below line
output_file = "output-image.png"

image = client.models.generate_images(
    model="imagen-4.0-generate-001",
    prompt="A dog reading a newspaper",
    config=GenerateImagesConfig(
        image_size="2K",
    ),
)

image.generated_images[0].image.save(output_file)

print(f"Created output image using {len(image.generated_images[0].image.image_bytes)} bytes")
# Example response:
# Created output image using 1234567 bytes

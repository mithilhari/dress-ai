import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Setup Environment
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Configuration
INPUT_IMAGE_PATH = "input_fashion.jpg" 
STYLES = ["Casual", "Formal", "Trendy"]

for m in client.models.list():
    if 'generate_images' in str(m).lower() or 'imagen' in m.name:
        print(f"Available Model: {m.name}")

def generate_fashion_styles():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: {INPUT_IMAGE_PATH} not found.")
        return

    # Load the base image
    input_image = Image.open(INPUT_IMAGE_PATH)

    for style in STYLES:
        print(f"\n--- Generating {style} Style ---")

        # STEP A: Gemini analyzes the photo and creates a text prompt
        analysis_prompt = (
            f"Analyze this person's pose and appearance. Write a detailed "
            f"fashion photography prompt to dress them in {style} attire. "
            f"Maintain their physical features and exact pose. "
            f"Output ONLY the prompt text."
        )
        
        describe_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[analysis_prompt, input_image]
        )
        
        refined_prompt = describe_response.text
        print(f"Imagen Prompt: {refined_prompt[:80]}...")

        # STEP B: Imagen generates the new image
        # FIXED: Use the specific model ID for AI Studio
        image_response = client.models.generate_images(
            model='imagen-4.0-generate-001', 
            prompt=refined_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1"
            )
        )

        # STEP C: Save the result
        if image_response.generated_images:
            output_filename = f"output_{style.lower()}.png"
            image_response.generated_images[0].image.save(output_filename)
            print(f"Successfully saved: {output_filename}")

if __name__ == "__main__":
    generate_fashion_styles()

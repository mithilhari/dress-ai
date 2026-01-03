import os
import io
import secrets
import zipfile
from flask import Flask, render_template, request, jsonify
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)

# CONFIGURATION: Replace with your Project ID
PROJECT_ID = "dress-ai-482502" 
LOCATION = "us-central1"

# Initialize Client for Vertex AI (Enterprise Mode)
# No API key needed; uses project credentials
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)

OUTPUT_DIR = os.path.join('static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_best_aspect_ratio(width, height):
    """Maps image dimensions to supported Imagen aspect ratios."""
    ratio = width / height
    if ratio > 1.5: return "16:9"
    if ratio > 1.1: return "4:3"
    if ratio < 0.6: return "9:16"
    if ratio < 0.85: return "3:4"
    return "1:1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    custom_style = request.form.get('custom_style', "").strip()
    
    styles = ["Casual", "Formal", "Trendy"]
    if custom_style:
        styles.append(custom_style)

    try:
        img_bytes = file.read()
        input_image = Image.open(io.BytesIO(img_bytes))
        
        # FIX: Detect original aspect ratio to prevent cropping
        w, h = input_image.size
        target_ratio = get_best_aspect_ratio(w, h)
        
        results = []
        file_paths = []

        for style in styles:
            # Step A: High-detail prompt generation
            analysis_prompt = (
                f"ACT AS A FASHION STYLIST. Analyze this person's pose and setting.\n"
                f"Write a 1-paragraph high-detail prompt to dress them in {style} attire.\n"
                f"Maintain EXACT pose and background. Framing: {target_ratio}."
            )
            
            describe_res = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[analysis_prompt, input_image]
            )

            gen_image = generate_image_with_imagen(
                prompt=describe_res.text.strip() + ", photorealistic, 8k",
                aspect_ratio=target_ratio
            )
             

            if gen_image.generated_images:
                fn = f"{style.lower()}_{secrets.token_hex(4)}.png"
                path = os.path.join(OUTPUT_DIR, fn)
                gen_image.save(path)
                results.append({"style": style, "url": f"/static/outputs/{fn}"})
                file_paths.append(path)

        # Step C: Bundle for download
        zip_fn = f"lookbook_{secrets.token_hex(3)}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_fn)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for p in file_paths:
                zipf.write(p, os.path.basename(p))

        return jsonify({"results": results, "zip_url": f"/static/outputs/{zip_fn}"})

    except Exception as e:
        return jsonify({"error": f"Vertex AI Error: {str(e)}"}), 500

def generate_image_with_imagen(prompt: str, aspect_ratio: str):
    """
    Generate a single image with Imagen 4.0.
    Returns a PIL.Image object.
    """
    image_res = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio
        )
    )

    if not image_res.generated_images:
        raise ValueError("Imagen returned no images")

    return image_res.generated_images[0].image

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

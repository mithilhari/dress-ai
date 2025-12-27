import os
import io
import secrets
import zipfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure static directories exist
OUTPUT_DIR = os.path.join('static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    custom_style = request.form.get('custom_style', "").strip()
    
    # Logic from your ai.py
    styles = ["Casual", "Formal", "Trendy"]
    if custom_style:
        styles.append(custom_style)

    try:
        img_bytes = file.read()
        input_image = Image.open(io.BytesIO(img_bytes))
        results = []
        file_paths = []

        for style in styles:
            # STEP A: Analysis logic from your ai.py
            analysis_prompt = (
                f"Analyze this person's pose and appearance. Write a detailed "
                f"fashion photography prompt to dress them in {style} attire. "
                f"Maintain their physical features and exact pose. Output ONLY text."
            )
            
            describe_res = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[analysis_prompt, input_image]
            )
            
            # STEP B: Imagen logic from your ai.py
            image_res = client.models.generate_images(
                model='imagen-4.0-generate-001', # Use stable version for reliability
                prompt=describe_res.text,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
            )

            if image_res.generated_images:
                fn = f"{style.lower()}_{secrets.token_hex(4)}.png"
                path = os.path.join(OUTPUT_DIR, fn)
                image_res.generated_images[0].image.save(path)
                
                results.append({"style": style, "url": f"/static/outputs/{fn}"})
                file_paths.append(path)

        # STEP C: Zip all images for the "Download All" button
        zip_fn = f"lookbook_{secrets.token_hex(3)}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_fn)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for p in file_paths:
                zipf.write(p, os.path.basename(p))

        return jsonify({"results": results, "zip_url": f"/static/outputs/{zip_fn}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Required for Cloud Run
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

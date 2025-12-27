import os
import io
import secrets
from flask import Flask, render_template, request, jsonify
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure output directory exists
os.makedirs(os.path.join('static', 'outputs'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image"}), 400
    
    file = request.files['image']
    custom_style = request.form.get('custom_style', "").strip()
    
    styles = ["Casual", "Formal", "Trendy"]
    if custom_style:
        styles.append(custom_style)

    try:
        img_bytes = file.read()
        input_image = Image.open(io.BytesIO(img_bytes))
        results = []

        for style in styles:
            # Step A: Gemini Analysis (from your original logic)
            analysis_prompt = (
                f"Analyze this person's pose and appearance. Write a detailed "
                f"fashion photography prompt to dress them in {style} attire. "
                f"Maintain their physical features and exact pose. Output ONLY the prompt text."
            )
            
            describe_res = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[analysis_prompt, input_image]
            )
            
            # Step B: Imagen Generation (from your original logic)
            image_res = client.models.generate_images(
                model='imagen-4.0-generate-001', 
                prompt=describe_res.text,
                config=types.GenerateImagesConfig(number_of_images=1)
            )

            if image_res.generated_images:
                fn = f"{style.lower()}_{secrets.token_hex(4)}.png"
                save_path = os.path.join('static', 'outputs', fn)
                image_res.generated_images[0].image.save(save_path)
                results.append({"style": style, "url": f"/static/outputs/{fn}"})
        
        return jsonify({"results": results})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Using 5001 as a backup in case 5000 is used by AirPlay on MacOS
    app.run(debug=True, port=5001)

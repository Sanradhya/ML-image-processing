from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from image_processing import process_image_with_pil, use_machine_learning_model

app = Flask(__name__)

# Define folders
IMAGE_FOLDER = 'images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Serve the upload form
@app.route('/')
def index():
    return render_template('upload.html')

# Serve static images from the images folder
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# Endpoint to process an uploaded image
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join(IMAGE_FOLDER, image_file.filename)

    # Save the uploaded image
    image_file.save(image_path)

    # Process the image with PIL
    processed_image = process_image_with_pil(image_path)

    # Use the ML model (optional)
    predictions = use_machine_learning_model(image_path)

    return jsonify({
        "message": "Image processed successfully.",
        "predictions": predictions.tolist(),
        "image_url": f"/images/{image_file.filename}"
    }), 200

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)

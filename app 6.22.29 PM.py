from flask import Flask , jsonify , request
import hashlib
import sqlite3
from PIL import Image
import numpy as np 
import os 
import cv2
import io
from keybert import KeyBERT


app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('image_data.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images 
                 (id INTEGER PRIMARY KEY, trace_id TEXT, description TEXT, image BLOB)''')
    conn.commit()
    conn.close()

create_table()

def generate_image_description(image):
    avg_color = np.array(image).mean(axis=(0, 1))
    return f"Image with average color: R={int(avg_color[0])}, G={int(avg_color[1])}, B={int(avg_color[2])}"

def generate_trace_id(image_bytes):
   
    return hashlib.sha256(image_bytes).hexdigest()

def apply_invisible_watermark(image, trace_id):
   
    watermark_text = trace_id[:10] 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


    text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    watermarked_image = cv2.putText(image, watermark_text, (text_x, text_y), font, font_scale, color, thickness)

    watermarked_image = Image.fromarray(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB))
    
    return watermarked_image

def save_image_data(trace_id, description, watermarked_image):
    
    image_bytes = io.BytesIO()
    watermarked_image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO images (trace_id, description, image) VALUES (?, ?, ?)", 
              (trace_id, description, image_bytes))
    conn.commit()
    conn.close()

def check_image_exists(trace_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM images WHERE trace_id=?", (trace_id,))
    result = c.fetchone()
    conn.close()
    return result

@app.route('/upload', methods=['POST'])
def upload_image():
    print(request)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        trace_id = generate_trace_id(image_bytes)
        
        if check_image_exists(trace_id):
            return jsonify({'message': f"Image already exists in the database with Trace ID: {trace_id}"}), 200
        else:
            description = generate_image_description(image)
            watermarked_image = apply_invisible_watermark(image, trace_id)
            save_image_data(trace_id, description, watermarked_image)
            return jsonify({'message': f"Image processed and saved with Trace ID: {trace_id}"}), 201


kw_model = KeyBERT()

@app.route('/keywords', methods=['POST'])
def keywords():
    if not request.is_json:
        return jsonify({'error': 'No JSON in request'}), 400
    
    data  = request.get_json()
    description = data.get('description','')

    if not description:
        description = """Hunza, nestled in the Karakoram Range, is an e
        nchanting valley with towering peaks, 
        verdant fields, and the serene Hunza River."""

    try: 
        keywords_with_scores = kw_model.extract_keywords(description, top_n=5)
        
        # Transform to JSON format with keywords as keys and true as values
        keywords = [keyword for keyword, score in keywords_with_scores]
        
        return jsonify({
            "data": keywords,
            "message": "Keywords successfully extracted"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
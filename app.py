from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import io
import os
import torch
import cv2  
import numpy as np
from tempfile import NamedTemporaryFile
from transformers import BlipProcessor, BlipForConditionalGeneration
import trimesh
from keybert import KeyBERT
import uuid
import requests
import boto3
from dotenv import load_dotenv
import requests


load_dotenv()

app = Flask(__name__)



model_path = "/Users/umairali/Documents/DafiLabs/Image-To-Text/local_blip_model"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
kw_model = KeyBERT()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_URL = os.getenv('AWS_ACCESS_URL')


# s3 connection
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

#trace id generation
def generate_trace_id():
    return str(uuid.uuid4())

#check duplication
def check_duplicate(file_bytes, trace_id):
    # Simulate the check: always return False for now (indicating no duplicate)
    # I will change this to True to simulate a duplicate
    return False

#Upload to s3
def upload_to_s3(file_bytes, trace_id):
    s3_key = f'{trace_id}.jpg'
    s3_client.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType='image/jpeg'
    )
    s3_url = f'{AWS_ACCESS_URL}/{s3_key}'
    return s3_url


"""
for now we are sending random url's which will be provided from nest.js API
"""
s3_url=  "https://d35rdqnaay08fm.cloudfront.net/6b60110c-f31b-4709-afd2-2452b9321060.jpg"
trace_id= "6b60110c-f31b-4709-afd2-2452b9321060"

#download image from s3
def download_image_from_s3(s3_url):
    """Download image from S3 given the URL."""
    try:
        response = requests.get(s3_url)
        if response.status_code == 200:
            return io.BytesIO(response.content), None
        else:
            return None, f"Failed to download image from S3, status code: {response.status_code}"
    except Exception as e:
        return None, str(e)


#apply invisible water mark 
#with showing water mark
# def apply_invisible_watermark(image_bytes, trace_id):
    """Apply invisible watermark with trace_id to the image."""
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Convert to numpy array
        image_np = np.array(image)
        
        # Define watermark properties
        watermark_text = trace_id[:10]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        # Convert to BGR for OpenCV processing
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
        text_x = (image_np.shape[1] - text_size[0]) // 2
        text_y = (image_np.shape[0] + text_size[1]) // 2
        watermarked_image_np = cv2.putText(image_np, watermark_text, (text_x, text_y), font, font_scale, color, thickness)

        # Convert back to RGB and PIL Image
        watermarked_image = Image.fromarray(cv2.cvtColor(watermarked_image_np, cv2.COLOR_BGR2RGB))
        
        # Save to a BytesIO object to return bytes
        buf = io.BytesIO()
        watermarked_image.save(buf, format='JPEG')
        buf.seek(0)  # Reset buffer position
        return buf.getvalue(), None

    except Exception as e:
        # Log the exception
        print(f"Exception: {e}")
        return None, str(e)


#Water mark image
def apply_invisible_watermark(image_bytes, trace_id):
    """Apply invisible watermark with trace_id using LSB to the image."""
    try:

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return None, f"Failed to open image: {str(e)}"
        image_np = np.array(image)
        watermark_binary = ''.join(format(ord(char), '08b') for char in trace_id[:10])  # Max 10 characters for watermark
        height, width, _ = image_np.shape
        if len(watermark_binary) > height * width:
            raise ValueError("Watermark is too large to fit in the image")
        binary_index = 0
        for row in range(height):
            for col in range(width):
                if binary_index < len(watermark_binary):
                    r = image_np[row, col, 0]
                    r_bin = format(r, '08b')
                    new_r_bin = r_bin[:-1] + watermark_binary[binary_index]
                    image_np[row, col, 0] = int(new_r_bin, 2)
                    binary_index += 1
                else:
                    break
            if binary_index >= len(watermark_binary):
                break


        watermarked_image = Image.fromarray(image_np)

        
        buf = io.BytesIO()
        watermarked_image.save(buf, format='JPEG')
        buf.seek(0) 
        return buf.getvalue(), None

    except Exception as e:
        # Log the exception
        print(f"Exception: {e}")
        return None, str(e)

        watermarked_image = Image.fromarray(image_np)

        buf = io.BytesIO()
        watermarked_image.save(buf, format='JPEG')
        buf.seek(0)  
        return buf.getvalue(), None

    except Exception as e:
        print(f"Exception: {e}")
        return None, str(e)

#Generate caption
def generate_caption(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        keywords = kw_model.extract_keywords(caption, top_n=5)
        keyword_list = [kw[0] for kw in keywords]  
        return caption, keyword_list
    except Exception as e:
        return f"Error generating caption: {str(e)}", []

#extract frames from videos for the model 
def extract_frames_from_video(video_bytes, frame_interval=30):
    try:
        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_file_path = temp_video_file.name

        frames = []
        video = cv2.VideoCapture(temp_video_file_path)
        frame_count = 0

        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG')
                frame_bytes = buf.getvalue()
                frames.append(frame_bytes)

            frame_count += 1

        video.release()
        os.remove(temp_video_file_path)

        return frames
    except Exception as e:
        return []


#3d process 3d object for model 
def process_3d_object(file_bytes):
    try:
        with NamedTemporaryFile(delete=False, suffix='.obj') as temp_3d_file:
            temp_3d_file.write(file_bytes)
            temp_3d_file_path = temp_3d_file.name

        mesh = trimesh.load(temp_3d_file_path)
        mesh_info = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'is_watertight': mesh.is_watertight
        }
        os.remove(temp_3d_file_path)
        return mesh_info
    except Exception as e:
        return {'error': str(e)}

#extract frames from GIf for the model 
def extract_frames_from_gif(gif_bytes, frame_interval=1):
    frames = []
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            if i % frame_interval == 0:
                buf = io.BytesIO()
                frame.convert("RGB").save(buf, format='JPEG')
                frame_bytes = buf.getvalue()
                frames.append(frame_bytes)
        return frames
    except Exception as e:
        return []

"""
This API will check the duplication of image it is exist or not we are sedning request to another API (nest js)
they will check out with (tace_id) and (s3_url) it the nest backend will respond us 
in the form of True and False
""" 
@app.route('/checkDuplication', methods=['POST'])
def check_duplication():
    file = request.files['file']
    trace_id = generate_trace_id()

    file_bytes = file.read()

   
    is_duplicate = False 

    if is_duplicate:
        return jsonify({'error': 'File is a duplicate.'}), 409  

    watermarked_image_bytes, error = apply_invisible_watermark(file_bytes, trace_id)
    if error:
        return jsonify({'error': f'Failed to apply watermark: {error}'}), 500

    s3_url = upload_to_s3(io.BytesIO(watermarked_image_bytes), trace_id)
    if not s3_url:
        return jsonify({'error': 'Failed to upload to S3'}), 500

    return jsonify({'trace_id': trace_id, 's3_url': s3_url}), 200



"""This is predict API endpoint for AI model"""
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Using the hardcoded S3 URL and trace_id for now
        print(f"Received trace_id: {trace_id}")  # Debug statement
        print(f"Received s3_url: {s3_url}")  # Debug statement
        
        # Download the image from the S3 URL
        image_stream, error = download_image_from_s3(s3_url)
        if error:
            return jsonify({'error': f'Failed to download image: {error}'}), 500
        
        # Perform prediction (generate caption and keywords)
        image_bytes = image_stream.read()
        caption, keywords = generate_caption(image_bytes)
        
        if 'Error' in caption:
            return jsonify({'error': caption}), 500
        
        return jsonify({ 'caption': caption, 'keywords': keywords , s3_url: "Image Url"}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True)

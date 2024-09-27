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
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allows all origins; customize as needed

'''Load the CLIP model and processor for NSFW detection'''

nsfw_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
nsfw_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


model_path = "/Users/umairali/Documents/DafiLabs/CrowdGen AI/AI/model"
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

'''Trace id generation'''
def generate_trace_id():
    return str(uuid.uuid4())

'''Function to check if an image is NSFW'''
def check_nsfw(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    inputs = nsfw_processor(text=["NSFW", "Safe"], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    nsfw_prob = probs[0][0].item()
    safe_prob = probs[0][1].item()

    if nsfw_prob > safe_prob:
        return "NSFW", nsfw_prob
    else:
        return "Safe", safe_prob


'''check duplication'''
def check_duplicate(file_bytes, trace_id):
    # Simulate the check: always return False for now (indicating no duplicate)
    # I will change this to True to simulate a duplicate
    return False

'''Upload to s3'''
def upload_to_s3(file_bytes, trace_id, file_extension, content_type):
    s3_key = f'{trace_id}{file_extension}'  # Use the correct file extension
    s3_client.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type  # Set the correct MIME type
    )
    s3_url = f'{AWS_ACCESS_URL}/{s3_key}'
    return s3_url


"""
for now we are sending random url's which will be provided from nest.js API
"""

# s3_url= "https://d35rdqnaay08fm.cloudfront.net/03ab3ad1-3923-46f7-863f-4b7fe1e0f771.jpg"
# trace_id = "03ab3ad1-3923-46f7-863f-4b7fe1e0f771"


'''download image from s3'''
def download_image_from_s3(s3_url):
    """Download image from S3 given the URL"""
    try:
        response = requests.get(s3_url)
        if response.status_code == 200:
            return io.BytesIO(response.content), None
        else:
            return None, f"Failed to download image from S3, status code: {response.status_code}"
    except Exception as e:
        return None, str(e)

'''Water mark'''
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

'''Generate caption'''
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

'''Extract frames from videos for the model'''
def extract_frames_from_video(video_bytes, frame_interval=30):
    try:
        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_file_path = temp_video_file.name

        frames = []
        results = []  # List to store captions for each frame
        video = cv2.VideoCapture(temp_video_file_path)
        frame_count = 0

        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Save the image to a buffer
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG')
                frame_bytes = buf.getvalue()

                # Generate a caption for the frame
                caption, keywords = generate_caption(frame_bytes)

                # Store the frame's caption and keywords
                results.append({
                    'frame_count': frame_count,
                    'caption': caption,
                    'keywords': keywords
                })

                # Store the frame bytes if needed later (optional)
                frames.append(frame_bytes)

            frame_count += 1

        # Clean up video file
        video.release()
        os.remove(temp_video_file_path)

        return {
            "frames_extracted": len(results),  # Number of frames processed
            "frame_captions": results  # Captions and keywords for each frame
        }
    except Exception as e:
        return {"error": str(e)}



'''3d process 3d object for model'''
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

"""Extract frames from GIF for the model"""
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
    try:
        file = request.files['file']
        trace_id = generate_trace_id()  # Generate a unique trace ID
        file_bytes = file.read()  # Read the file content as bytes
        file_type = file.content_type  # Detect file type

        # Default file extension and content type
        file_extension = ''
        content_type = file_type

        # Handle images
        if 'image' in file_type:
            if 'gif' in file_type:  # Check if it's a GIF
                file_extension = '.gif'
                '''Step 1: Extract frames from GIF'''
                gif_frames = extract_frames_from_gif(file_bytes)

                '''Step 2: Check duplication for GIF frames'''
                is_duplicate = check_duplicate(gif_frames, trace_id)
                if is_duplicate:
                    return jsonify({
                        'error': 'GIF is a duplicate.',
                        "isDuplicate": True,
                    }), 409  # Conflict

            else:  
                if file_extension in {'png', 'jpg', 'jpeg', 'gif'}:

                    '''Step 1: Check if the image is NSFW'''
                    nsfw_status, prob = check_nsfw(file_bytes)
                    if nsfw_status == "NSFW":
                        return jsonify({
                            'error': 'This content may violate our usage policies.',
                            "Violate": True
                        }), 403

                '''Step 2: Check duplication for images'''
                is_duplicate = check_duplicate(file_bytes, trace_id)
                if is_duplicate:
                    return jsonify({
                        'error': 'File is a duplicate.',
                        "isDuplicate": True,
                    }), 409  

                '''Step 3: Apply watermark for images'''
                watermarked_image_bytes, error = apply_invisible_watermark(file_bytes, trace_id)
                if error:
                    return jsonify({
                        'error': f'Failed to apply watermark: {error}'
                    }), 500

                # Use watermarked bytes for the final upload
                file_bytes = watermarked_image_bytes

        # Handle videos
        elif 'video' in file_type:
            file_extension = '.mp4'  # Default to MP4; adjust based on MIME type if needed

            '''Step 1: Extract frames from the video for comparison'''
            video_frames = extract_frames_from_video(file_bytes)

            '''Step 2: Check duplication for videos using frames'''
            is_duplicate = check_duplicate(video_frames, trace_id)
            if is_duplicate:
                return jsonify({
                    'error': 'Video is a duplicate.',
                    "isDuplicate": True,
                }), 409  # Conflict

        # Handle 3D objects (e.g., .obj files)
        elif file.filename.endswith('.obj'):
            file_extension = '.obj'
            content_type = 'application/octet-stream'  # MIME type for 3D objects

            '''Step 1: Process 3D object'''
            object_info = process_3d_object(file_bytes)

            '''Step 2: Check duplication for 3D objects'''
            is_duplicate = check_duplicate(file_bytes, trace_id)
            if is_duplicate:
                return jsonify({
                    'error': '3D Object is a duplicate.',
                    "isDuplicate": True,
                }), 409  # Conflict

        else:
            return jsonify({
                'error': 'Unsupported file type.'
            }), 400

        '''Step 4: Upload the file to S3 with the correct extension and MIME type'''
        s3_url = upload_to_s3(file_bytes, trace_id, file_extension, content_type)
        if not s3_url:
            return jsonify({
                'error': 'Failed to upload to S3'
            }), 500

        '''Step 5: Return success response with trace_id and S3 URL'''
        return jsonify({
            'trace_id': trace_id,
            's3_url': s3_url,
            "isDuplicate": False
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500




def determine_file_type(s3_url):
    if s3_url.endswith('.jpg') or s3_url.endswith('.jpeg') or s3_url.endswith('.png'):
        return 'image'
    elif s3_url.endswith('.gif'):
        return 'gif'
    elif s3_url.endswith('.mp4'):
        return 'mp4'
    elif s3_url.endswith('.obj'):
        return 'obj'
    else:
        return 'unknown'



"""This is predict API endpoint for AI model"""
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the S3 URL and trace_id from the request
        data = request.get_json()
        trace_id = data.get('trace_id')
        s3_url = data.get('s3_url')

        if not trace_id or not s3_url:
            return jsonify({'error': 'trace_id and s3_url are required'}), 400

        '''Step 1: Download the file from S3'''
        file_stream, error = download_image_from_s3(s3_url)
        if error:
            return jsonify({'error': f'Failed to download file from S3: {error}'}), 500

        # Detect the file type by examining the file extension in the URL
        file_type = determine_file_type(s3_url)

        '''Step 2: Handle each file type accordingly'''
        if 'image' in file_type:  # Handle images (JPEG, PNG)
            '''Generate caption for the image'''
            file_bytes = file_stream.read()
            caption, keywords = generate_caption(file_bytes)
            if 'Error' in caption:
                return jsonify({'error': caption}), 500

            # Return the response with trace_id, caption, and keywords
            return jsonify({
                'trace_id': trace_id,
                'caption': caption,
                'keywords': keywords
            }), 200

        elif 'gif' in file_type:  # Handle GIFs
            '''Extract frames from GIF'''
            gif_bytes = file_stream.read()
            gif_frames = extract_frames_from_gif(gif_bytes)
            if not gif_frames:
                return jsonify({'error': 'Failed to extract frames from GIF'}), 500
            
            # Process captions from GIF frames (assuming extract_frames_from_gif returns captions)
            gif_captions = [frame['caption'] for frame in gif_frames]
            all_captions = ' '.join(gif_captions)  # Join captions into a single string
            all_keywords = [kw for frame in gif_frames for kw in frame['keywords']]

            return jsonify({
                'trace_id': trace_id,
               
                'caption': all_captions,
                'keywords': all_keywords
            }), 200

        elif 'mp4' in file_type:  # Handle .mp4 videos
            '''Extract frames from the video'''
            video_bytes = file_stream.read()
            extract_response = extract_frames_from_video(video_bytes)
            if 'error' in extract_response:
                return jsonify({"error": extract_response['error']}), 500

            # Extract captions and keywords into separate variables
            captions = ' '.join(item['caption'] for item in extract_response['frame_captions'])
            keywords = [kw for item in extract_response['frame_captions'] for kw in item['keywords']]

            # Return trace_id and consolidated lists of captions and keywords
            return jsonify({
                "trace_id": trace_id,
                "caption": captions,  # Combine all captions into a single string
                "keywords": keywords  # Return keywords as an array
            }), 200

        elif 'obj' in file_type:  # Handle 3D objects
            '''Process 3D object'''
            object_bytes = file_stream.read()
            object_info = process_3d_object(object_bytes)
            if 'error' in object_info:
                return jsonify({'error': object_info['error']}), 500

            return jsonify({'trace_id': trace_id, '3D_object_info': object_info}), 200

        else:
            return jsonify({'error': 'Unsupported file type'}), 400

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    


@app.route('/hello', methods=['GET']) 
def hello():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(debug=True)


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
from transformers import CLIPProcessor, CLIPModel , Speech2TextProcessor, Speech2TextForConditionalGeneration
from io import BytesIO
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import hashlib
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import librosa
from PyPDF2 import PdfReader
# from docx import Document

# audio_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
# audio_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")





load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True) 


'''Models'''
nsfw_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
nsfw_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

audio_model_path = "/Users/umairali/Documents/DafiLabs/CrowdGen AI/AI/audio-model"
audio_processor = Speech2TextProcessor.from_pretrained(audio_model_path)
audio_model = Speech2TextForConditionalGeneration.from_pretrained(audio_model_path)

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



'''s3 connection'''
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

'''Trace id generation'''
def generate_trace_id(file_bytes):
    
    return hashlib.sha256(file_bytes).hexdigest()


def transcribe_audio(audio_file_bytes):
    """Transcribe audio to text."""
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_file_bytes), sr=16000)
        inputs = audio_processor(audio_array, sampling_rate=16000, return_tensors="pt")
        generated_ids = audio_model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = audio_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0] 
    except Exception as e:
        print(f"Error in audio transcription: {str(e)}")
        return None

'''Function to check if an image is NSFW'''
def check_nsfw(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")  
        inputs = nsfw_processor(text=["NSFW", "Safe"], images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = nsfw_model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        nsfw_prob = probs[0][0].item()
        safe_prob = probs[0][1].item()

        print(f"NSFW Probability: {nsfw_prob}, Safe Probability: {safe_prob}") 

        if nsfw_prob > 0.85:
            return "NSFW", nsfw_prob
        else:
            return "Safe", safe_prob
    except Exception as e:
        print(f"Error processing NSFW check: {e}")  
        return "Error", None 

'''check duplication'''
def check_duplicate(trace_id):
    """Check if the trace ID already exists using an external API."""
    try:
        url = f'https://533e-2400-adc5-195-8700-c4c7-dbc1-39fc-2e99.ngrok-free.app/assets/check-duplication/{trace_id}'
        response = requests.get(url, headers={'accept': '*/*'})
        if response.status_code == 200:
            is_duplicate = response.json() 
            return is_duplicate
        else:
            print(f"Error: Received status code {response.status_code} from the API.")
            return False  
    except Exception as e:
        print(f"Exception occurred while checking duplication: {str(e)}")
        return False  

'''Upload to s3'''
def upload_to_s3(file_bytes, trace_id, file_extension, content_type):
    s3_key = f'{trace_id}{file_extension}'
    s3_client.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type
    )
    s3_url = f'{AWS_ACCESS_URL}/{s3_key}'
    return s3_url



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
        watermark_binary = ''.join(format(ord(char), '08b') for char in trace_id[:10])  
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
        results = [] 
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

                caption, keywords = generate_caption(frame_bytes)
                results.append({
                    'frame_count': frame_count,
                    'caption': caption,
                    'keywords': keywords
                })
                frames.append(frame_bytes)

            frame_count += 1
        video.release()
        os.remove(temp_video_file_path)

        return {
            "frames_extracted": len(results), 
            "frame_captions": results 
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

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return jsonify({'error': 'File size exceeds the maximum limit.'}), 413  







'''This is only for pdf to keyword not here we used any AI model here is used Keyberd library'''
def extract_keywords_from_pdf(pdf_bytes):
    """Extract keywords from a PDF file using KeyBERT."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'  
        keywords = kw_model.extract_keywords(text, top_n=5)
        return [kw[0] for kw in keywords] 
    except Exception as e:
        print(f"Error extracting keywords from PDF: {str(e)}")
        return []


'''This function for txt any text file'''
def extract_keywords_from_txt(txt_bytes):
    """Extract keywords from a TXT file using KeyBERT."""
    try:
        text = txt_bytes.decode('utf-8')

        keywords = kw_model.extract_keywords(text, top_n=5)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error extracting keywords from TXT: {str(e)}")
        return []



'''this is for word docx file'''
def extract_keywords_from_docx(docx_bytes):
    """Extract keywords from a DOCX file using KeyBERT."""
    try:
        doc = Document(io.BytesIO(docx_bytes)) 
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n' 

        keywords = kw_model.extract_keywords(text, top_n=5)  
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error extracting keywords from DOCX: {str(e)}")
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
        file_bytes = file.read() 
        file_type = file.content_type 
        file_extension = os.path.splitext(file.filename)[1] 
        print("Received content type:", file.content_type)
        trace_id = generate_trace_id(file_bytes)
        if 'image' in file_type:
            nsfw_status, prob = check_nsfw(file_bytes)
            if nsfw_status == "NSFW":
                return jsonify({
                    'error': 'This content may violate our usage policies.',
                    "violate": True
                }), 403
            is_duplicate = check_duplicate(trace_id)
            if is_duplicate:
                return jsonify({
                    'error': 'File is a duplicate.',
                    "isDuplicate": True,
                }), 409  

        elif 'video' in file_type:  
            nsfw_status, prob = check_nsfw(file_bytes)
            if nsfw_status == "NSFW":
                return jsonify({
                    'error': 'This content may violate our usage policies.',
                    "violate": True
                }), 403

            video_frames = extract_frames_from_video(file_bytes)
            is_duplicate = check_duplicate(trace_id)
            if is_duplicate:
                return jsonify({
                    'error': 'Video is a duplicate.',
                    "isDuplicate": True,
                }), 409 

        elif 'gif' in file_type: 
            nsfw_status, prob = check_nsfw(file_bytes)
            if nsfw_status == "NSFW":
                return jsonify({
                    'error': 'This content may violate our usage policies.',
                    "violate": True
                }), 403

            gif_frames = extract_frames_from_gif(file_bytes)
            if not gif_frames:
                return jsonify({'error': 'Failed to extract frames from GIF'}), 500
            
            is_duplicate = check_duplicate(trace_id)
            if is_duplicate:
                return jsonify({
                    'error': 'GIF is a duplicate.',
                    "isDuplicate": True,
                }), 409 

        elif file.filename.endswith('.obj'): 
            nsfw_status, prob = check_nsfw(file_bytes)
            if nsfw_status == "NSFW":
                return jsonify({
                    'error': 'This content may violate our usage policies.',
                    "violate": True
                }), 403
            
            object_info = process_3d_object(file_bytes)
            is_duplicate = check_duplicate(trace_id)
            if is_duplicate:
                return jsonify({
                    'error': '3D Object is a duplicate.',
                    "isDuplicate": True,
                }), 409  
        elif 'audio' in file_type:
            nsfw_status, prob = check_nsfw(file_bytes) 
            if nsfw_status == "NSFW":
                return jsonify({
                    'error': 'This content may violate our usage policies.',
                    "violate": True
                }), 403
            s3_url = upload_to_s3(file_bytes, trace_id, file_extension, file_type)
            if not s3_url:
                return jsonify({'error': 'Failed to upload to S3'}), 500

            return jsonify({
                'trace_id': trace_id,
                's3_url': s3_url,
                "isDuplicate": False
            }), 200
        


        elif file.filename.endswith('.pdf') or file.filename.endswith('.docx') or file.filename.endswith('.txt'):
            s3_url = upload_to_s3(file_bytes, trace_id, file_extension, file_type)
            if not s3_url:
                return jsonify({'error': 'Failed to upload to S3'}), 500

            return jsonify({'trace_id': trace_id, 's3_url': s3_url, "isDuplicate": False}), 200
        
        else:
            return jsonify({
                'error': 'Unsupported file type.'
            }), 400

        s3_url = upload_to_s3(file_bytes, trace_id, file_extension, file_type)
        if not s3_url:
            return jsonify({
                'error': 'Failed to upload to S3'
            }), 500

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
    elif s3_url.endswith('.wav') or s3_url.endswith('.mp3') or s3_url.endswith('M4A') or s3_url.endswith('DSD') or s3_url.endswith('FLAC') or s3_url.endswith('OGG'):
        return 'audio' 
    elif s3_url.endswith('.pdf') or s3_url.endswith('.docx') or s3_url.endswith('.txt'):
        return 'pdf'
    else:
        return 'unknown'




"""This is predict API endpoint for AI model"""
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        trace_id = data.get('trace_id')
        s3_url = data.get('s3_url')

        if not trace_id or not s3_url:
            return jsonify({'error': 'trace_id and s3_url are required'}), 400

        file_stream, error = download_image_from_s3(s3_url)
        if error:
            return jsonify({'error': f'Failed to download file from S3: {error}'}), 500

        file_type = determine_file_type(s3_url)
        if 'image' in file_type: 
            '''Generate caption for the image'''
            file_bytes = file_stream.read()
            caption, keywords = generate_caption(file_bytes)
            if 'Error' in caption:
                return jsonify({'error': caption}), 500
            return jsonify({
                'trace_id': trace_id,
                'caption': caption,
                'keywords': keywords
            }), 200

        elif 'gif' in file_type: 
            gif_bytes = file_stream.read()
            gif_frames = extract_frames_from_gif(gif_bytes)
            if not gif_frames:
                return jsonify({'error': 'Failed to extract frames from GIF'}), 500
            
            gif_captions = [frame['caption'] for frame in gif_frames]
            all_captions = ' '.join(gif_captions)  
            all_keywords = [kw for frame in gif_frames for kw in frame['keywords']]

            return jsonify({
                'trace_id': trace_id,
                'caption': all_captions,
                'keywords': all_keywords
            }), 200

        elif 'mp4' in file_type:
            '''Extract frames from the video'''
            video_bytes = file_stream.read()
            extract_response = extract_frames_from_video(video_bytes)
            if 'error' in extract_response:
                return jsonify({"error": extract_response['error']}), 500

            captions = ' '.join(item['caption'] for item in extract_response['frame_captions'])
            keywords = [kw for item in extract_response['frame_captions'] for kw in item['keywords']]

        
            return jsonify({
                "trace_id": trace_id,
                "caption": captions, 
                "keywords": keywords 
            }), 200

        elif 'obj' in file_type: 
            '''Process 3D object'''
            object_bytes = file_stream.read()
            object_info = process_3d_object(object_bytes)
            if 'error' in object_info:
                return jsonify({'error': object_info['error']}), 500

            return jsonify({'trace_id': trace_id, '3D_object_info': object_info}), 200


        elif 'audio' in file_type:
            audio_bytes = file_stream.read()
            transcription = transcribe_audio(audio_bytes)
            if transcription is None:
                return jsonify({'error': 'Error during transcription'}), 500

            keywords = kw_model.extract_keywords(transcription, top_n=5)
            keyword_list = [kw[0] for kw in keywords]  
            return jsonify({
                'trace_id': trace_id,
                'caption': transcription,
                'keywords': keyword_list,  
                'isDuplicate': False
            }), 200
        

        elif 'pdf' in file_type:
            pdf_bytes = file_stream.read()
            keywords = extract_keywords_from_pdf(pdf_bytes)
            return jsonify({
                'trace_id': trace_id,
                'keywords': keywords,
            }), 200
        
        elif file_type == 'docx':
            docx_bytes = file_stream.read()
            keywords = extract_keywords_from_docx(docx_bytes)
            return jsonify({
                'trace_id': trace_id,
                'keywords': keywords,
            }), 200
        
        elif file_type == 'txt':
            txt_bytes = file_stream.read()
            keywords = extract_keywords_from_txt(txt_bytes)
            return jsonify({
                'trace_id': trace_id,
                'keywords': keywords,
            }), 200
        
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)


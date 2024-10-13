# CrowdGenAI

This project is a Flask-based API that integrates several AI models and services for processing multimedia files, including images, videos, audio, PDFs, and text files. The API provides functionalities like NSFW image detection, audio transcription, caption and keyword extraction, watermarking, and file duplication checking using cloud storage (AWS S3).

## Features

- **NSFW Image Detection**: Classifies images as NSFW or Safe using the OpenAI CLIP model.
- **Audio Transcription**: Transcribes audio files to text using Speech2Text models.
- **File Duplication Checking**: Checks if a file is a duplicate by generating a unique trace ID and sending it to an external API.
- **Caption and Keyword Extraction**:
  - Images: Extracts captions and keywords using the BLIP model.
  - Videos: Extracts frames, generates captions, and retrieves keywords from each frame.
  - PDFs, DOCX, TXT: Extracts keywords and caption text using KeyBERT.
- **Invisible Watermarking**: Applies an invisible watermark using LSB (Least Significant Bit) encoding.
- **S3 Integration**: Uploads and retrieves files from an AWS S3 bucket.
- **3D Object Processing**: Extracts metadata from 3D object files (e.g., OBJ format).
- **GIF and Video Frame Extraction**: Extracts frames from GIF and video files at specified intervals for further processing.
- **GPT-2 Text Generation**: Provides text generation from a given prompt using GPT-2.

## Technologies and Libraries

- **Flask**: Web framework used to create the API.
- **PIL (Pillow)**: Used for image processing.
- **Torch (PyTorch)**: Powers AI models like CLIP, BLIP, and GPT-2.
- **KeyBERT**: Extracts keywords from text.
- **librosa**: Used for audio processing.
- **boto3**: AWS SDK for Python to interact with S3.
- **OpenAI CLIP**: For image and text similarity.
- **BlipProcessor and BlipForConditionalGeneration**: For image captioning.
- **Speech2TextForConditionalGeneration**: For audio-to-text conversion.
- **Trimesh**: For processing 3D object files.
- **PDFReader (PyPDF2)**: Extracts text from PDFs.
- **docx (python-docx)**: Processes DOCX files.

## Setup and Installation

## Prerequisites

- Python 3.7 or higher
- AWS credentials for S3 access (with bucket permissions)
- GPU-enabled environment (for faster model inference)

## Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/umairchanna57/CrowdGenAI.git
   cd CrowdGenAI


2. Create a virtual environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Install the required dependencies
    ```bash
    pip install -r requirements.txt


4. Run the Flask app
    ```bash 
    python app.py


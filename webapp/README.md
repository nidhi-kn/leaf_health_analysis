# Tomato Disease Detection Web Application

A beautiful, modern web interface for AI-powered tomato disease detection.

## Features

- 🎨 **Modern UI** - Beautiful gradient design with smooth animations
- 📸 **Drag & Drop** - Easy image upload with drag-and-drop support
- 🤖 **AI-Powered** - MobileNetV2 model with 86.12% accuracy
- 📊 **Top-3 Predictions** - Shows confidence scores for multiple predictions
- 📱 **Responsive** - Works on desktop, tablet, and mobile devices
- ⚡ **Fast** - Quick inference with optimized model

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open in Browser

Navigate to: `http://localhost:5000`

## Usage

1. **Upload Image**: Click the upload area or drag & drop a tomato leaf image
2. **Analyze**: Click the "Analyze Leaf" button
3. **View Results**: See the AI diagnosis with confidence scores
4. **Repeat**: Analyze another image with one click

## Supported Image Formats

- JPG/JPEG
- PNG
- Maximum file size: 16MB

## API Endpoints

### `GET /`
Home page with upload interface

### `POST /predict`
Upload image for disease prediction

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "success": true,
  "predicted_class": "Late Blight",
  "confidence": 94.5,
  "top_3_predictions": [
    {"class": "Late Blight", "confidence": 94.5},
    {"class": "Early Blight", "confidence": 3.2},
    {"class": "Healthy", "confidence": 1.1}
  ],
  "image": "data:image/jpeg;base64,...",
  "model_info": {...}
}
```

### `GET /about`
About page with project information

### `GET /model-info`
Get model information and available classes

## Deployment

### Local Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
docker build -t tomato-disease-detector .
docker run -p 5000:5000 tomato-disease-detector
```

## Project Structure

```
webapp/
├── app.py                 # Flask application
├── templates/
│   ├── index.html        # Main upload page
│   └── about.html        # About page
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Information

- **Architecture**: MobileNetV2
- **Accuracy**: 86.12%
- **Type**: CNN (Convolutional Neural Network)
- **Input Size**: 224x224 RGB images
- **Classes**: 10 tomato disease categories

## Technologies Used

- **Backend**: Flask (Python)
- **AI Model**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL (Pillow)

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

This project is part of a research study on CNN vs Transformer architectures for agricultural computer vision.

## Contact

For questions or issues, please refer to the main project documentation.

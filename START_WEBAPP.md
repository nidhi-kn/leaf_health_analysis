# 🚀 Quick Start Guide - Web Application

## Your Web App is Ready!

A beautiful, modern web interface for your tomato disease detection AI model.

## 📁 What Was Created

```
webapp/
├── app.py                    # Flask backend server
├── templates/
│   ├── index.html           # Main upload page (beautiful UI)
│   └── about.html           # About/info page
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
├── run.bat                 # Windows startup script
└── run.sh                  # Linux/Mac startup script
```

## 🎯 Features

✅ **Beautiful Modern UI** - Gradient design with smooth animations
✅ **Drag & Drop Upload** - Easy image upload
✅ **Real-time AI Analysis** - Instant disease detection
✅ **Top-3 Predictions** - Shows confidence scores
✅ **Responsive Design** - Works on all devices
✅ **Production Ready** - Uses your 86.12% accuracy model

## 🏃 How to Start

### Option 1: Windows (Easy)
```bash
cd webapp
run.bat
```

### Option 2: Command Line
```bash
cd webapp
pip install -r requirements.txt
python app.py
```

### Option 3: Linux/Mac
```bash
cd webapp
chmod +x run.sh
./run.sh
```

## 🌐 Access the App

Once started, open your browser and go to:
```
http://localhost:5000
```

## 📸 How to Use

1. **Upload Image**: Click or drag & drop a tomato leaf photo
2. **Analyze**: Click "Analyze Leaf" button
3. **View Results**: See AI diagnosis with confidence scores
4. **Repeat**: Analyze another image

## 🎨 What You'll See

### Main Page Features:
- Beautiful gradient purple background
- Large upload area with drag & drop
- Real-time image preview
- AI diagnosis with confidence percentage
- Top 3 predictions with scores
- Model information badges

### Results Display:
- Uploaded image preview
- Main prediction (disease name + confidence)
- Top 3 alternative predictions
- Clean, professional layout

## 🔧 Troubleshooting

### If model not found:
The app looks for the model at: `../deployment/tomato_disease_model.pkl`

Make sure your deployment folder is in the parent directory.

### If port 5000 is busy:
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change to 8080
```

### If dependencies missing:
```bash
pip install flask tensorflow pillow numpy
```

## 📱 Mobile Friendly

The web app is fully responsive and works great on:
- Desktop computers
- Tablets
- Mobile phones

## 🚀 Deploy to Production

### Using Gunicorn (Recommended):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker:
Create a `Dockerfile` in the webapp directory:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

Then:
```bash
docker build -t tomato-detector .
docker run -p 5000:5000 tomato-detector
```

## 🎯 API Usage

You can also use the API programmatically:

```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('leaf_image.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

## 📊 Model Information

The web app uses your trained model:
- **Model**: MobileNetV2
- **Accuracy**: 86.12%
- **Architecture**: CNN
- **Classes**: 10 tomato diseases

## 🎉 You're All Set!

Your professional web application is ready to use. Just run it and start detecting tomato diseases with AI!

---

**Need help?** Check the `webapp/README.md` for detailed documentation.

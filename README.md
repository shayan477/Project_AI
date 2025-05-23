🔥 Real-Time Human Action Recognition with Vision Transformers
A deep learning project that detects fighting vs non-fighting actions in videos using a custom-built Vision Transformer (ViT) model. The system performs real-time video classification through a clean and interactive Streamlit web app, making it ideal for safety and surveillance applications.

📁 Project Structure
.
├── app.py                    # Streamlit web app
├── demo.py                   # CLI demo for image testing
├── frame_extraction.py       # Utility to extract frames from videos
├── human_action_predictor.py # ViT model + prediction logic
├── vit_model.pth             # Trained model checkpoint
├── AI_Project.ipynb          # Training notebook
└── README.md

🧠 Model Architecture
Backbone: Custom Vision Transformer (ViT)

Input: 224x224 RGB images

Classes: fighting, non-fighting

Training: Conducted using PyTorch on labeled frame datasets extracted from videos

🖼️ Sample Results
Uploaded videos are processed frame-by-frame with color-coded bounding:

🟥 Fighting Detected

🟩 No Fighting

🛠️ How to Use
1. Clone the Repository

git clone https://github.com/yourusername/human-action-recognition-vit.git
cd human-action-recognition-vit

2. Create and Activate a Virtual Environment
For Windows:
python -m venv venv
venv\Scripts\activate

For macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Install Requirements
pip install -r requirements.txt

4. Run the App
streamlit run app.py

5. Upload a Video
Use the sidebar in the app to upload a video and adjust:

Model path

Device (cpu or cuda)

Confidence threshold

Frame skip interval


🏋️ Training
To train your own model:

Use frame_extraction.py to extract frames from labeled videos.

Use AI_Project.ipynb to train a ViT model on the extracted frames.


Save the model as vit_model.pth and use it with the app.

📦 Sample Usage (Command Line)
python demo.py --model vit_model.pth --image test.jpg --device cpu

📌 Requirements
Python 3.7+

PyTorch

torchvision

Streamlit

OpenCV

Pillow


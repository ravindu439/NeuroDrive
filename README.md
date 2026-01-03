# NeuroDrive

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**NeuroDrive** is a production-ready vehicle detection and classification system powered by YOLOv8. It identifies 8 distinct vehicle types with high accuracy, featuring a modern Streamlit interface for both real-time inference and batch processing.

## 🚀 Key Features

* **Advanced Detection**: Identifies Cars, Buses, Trucks, Motorcycles, Bicycles, Three-wheelers, Tractors, and Vans.
* **Batch Processing**: Automated analysis of image folders with dominant class categorization.
* **Analytics Dashboard**: Visual reports, class distribution charts, and downloadable CSV summaries.
* **Professional UI**: Glassmorphism-inspired design with dark mode and smooth animations.

## 🛠️ Tech Stack

* **Model**: YOLOv8 (Ultralytics) - Large/Medium backbones.
* **Interface**: Streamlit - Interactive web application.
* **Processing**: OpenCV & Pandas - Image manipulation and data analysis.
* **Training**: PyTorch - Custom training pipeline on 15k+ vehicle images.

## 🏁 Quick Start

### 1. Installation

**Windows** (PowerShell)

```powershell
git clone https://github.com/yourusername/NeuroDrive.git
cd NeuroDrive
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

**Linux / macOS**

```bash
git clone https://github.com/yourusername/NeuroDrive.git
cd NeuroDrive
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup

The inference app lives in `VehicleDetectionSystem`. You need a trained model to run it.

1. **Train a model** (Optional, if you don't have one):
    Open `training.ipynb` and run the cells to train YOLOv8 on your dataset.

2. **Deploy Model**:
    Copy your `best.pt` to the app's model directory.

    **Windows**

    ```powershell
    copy runs\detect\train\weights\best.pt VehicleDetectionSystem\models\best.pt
    ```

    **Linux / macOS**

    ```bash
    cp runs/detect/train/weights/best.pt VehicleDetectionSystem/models/best.pt
    ```

### 3. Run Application

**Windows**

```powershell
cd VehicleDetectionSystem
pip install -r requirements.txt
streamlit run app.py
```

**Linux / macOS**

```bash
cd VehicleDetectionSystem
pip install -r requirements.txt
streamlit run app.py
```

Access the app at `http://localhost:8501`.

## 📁 Project Structure

```
NeuroDrive/
├── training.ipynb              # Reproducible training pipeline
├── requirements.txt            # Training dependencies
├── datasets/                   # (Ignored) Training data
├── runs/                       # (Ignored) Training artifacts
└── VehicleDetectionSystem/     # Main Application
    ├── app.py                  # Streamlit entry point
    ├── style.css               # Modern UI styling
    ├── models/                 # Inference weights
    └── utils/                  # Core detection logic
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

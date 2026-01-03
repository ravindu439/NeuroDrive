# 🚗 Vehicle Detection System

A modern, professional Streamlit application for detecting and classifying vehicles using YOLOv8 deep learning model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🌟 Features

- **Single Image Detection**: Analyze individual images with detailed statistics
- **Batch Processing**: Process multiple images at once with automated organization
- **8 Vehicle Classes**: Detects bicycle, bus, car, motorcycle, three_wheeler, tractor, truck, van
- **Professional UI**: Clean blue color scheme with Inter/Roboto fonts
- **External CSS**: Easy customization through style.css
- **Report Generation**: Automatic CSV report creation for batch processing
- **Class Organization**: Automatically organizes results by dominant vehicle class
- **Confidence Control**: Adjustable detection threshold for precision tuning
- **Horizontal Bar Charts**: Improved visualization with readable class names
- **Logo Support**: Built-in logo integration system (add your logo to assets/)
- **Export Options**: Download annotated images and comprehensive reports

## 📋 Requirements

- Python 3.8 or higher
- YOLOv8 model weights (best.pt)
- Required packages (see requirements.txt)

## 🚀 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd VehicleDetectionSystem
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Model Weights

Place your trained YOLOv8 model file in the `models/` directory:

**Option 1: Copy from existing training run**
```bash
cp ../runs/detect/vehicle_detector3/weights/best.pt models/best.pt
```

**Option 2: Manual placement**
- Copy your `best.pt` file to the `models/` folder
- Make sure the file is named exactly `best.pt`

## 📂 Project Structure

```
VehicleDetectionSystem/
├── app.py                      # Main Streamlit application (410 lines)
├── style.css                   # External CSS styling (441 lines)
├── models/                     # Model weights directory
│   └── best.pt                # YOLOv8 trained model (22MB)
├── assets/                     # Logo and icons directory
│   └── logo.png               # Your logo (place here)
├── results/                    # Output directory for detections
│   ├── batch/                 # Organized by class (for folder mode)
│   └── report.csv             # Generated report (for folder mode)
├── uploads/                    # Temporary storage for uploaded files
├── utils/                      # Helper modules
│   ├── __init__.py            # Package initialization
│   ├── detector.py            # Vehicle detection logic (8 classes)
│   ├── reporter.py            # Report generation
│   └── image_helper.py        # Logo and icon helper functions
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation (this file)
├── QUICKSTART.txt             # Quick start guide
├── check_setup.py             # Verify installation
└── test_detector.py           # Test detector module
```

## 💻 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Single Image Mode

1. Select **"Single Image"** in the sidebar
2. Adjust the **confidence threshold** (default: 0.25)
3. Click **"Browse files"** and upload an image
4. Click **"🚀 Classify Vehicles"** button
5. View the annotated image and detection statistics
6. Download the result using the download button

### Folder Mode

1. Select **"Folder"** in the sidebar
2. Adjust the **confidence threshold** as needed
3. Upload **multiple images** at once
4. Click **"🚀 Classify Vehicles"** button
5. View overall statistics and class distribution
6. Check the detailed report table
7. Download the CSV report
8. Find organized results in `results/batch/` (sorted by vehicle class)

## 🎨 UI Features

- **Professional Blue Theme**: Navy and light blue color scheme (#0b3b79, #4a90e2, #2c7be5)
- **Enhanced Sidebar**: Gradient navy background with improved visibility
- **External CSS**: All styling in style.css for easy customization
- **Professional Fonts**: Inter and Roboto (non-AI generated look)
- **Responsive Layout**: Works on different screen sizes
- **Interactive Cards**: Statistics displayed in beautiful rounded cards
- **Progress Indicators**: Real-time progress bars for batch processing
- **Color-Coded Detections**: 8 unique colors per vehicle class
- **Horizontal Bar Charts**: Class names displayed horizontally for better readability
- **Logo Integration**: Built-in system for adding your logo and icons
- **Smooth Animations**: Hover effects and transitions for better UX

## 📊 Supported Vehicle Classes

The model can detect the following vehicle types (8 classes with unique colors):

| Class | Color | Bounding Box |
|-------|-------|--------------|
| � **Bicycle** | Orange | `(255, 165, 0)` |
| 🚌 **Bus** | Magenta | `(255, 0, 255)` |
| � **Car** | Green | `(0, 255, 0)` |
| 🏍️ **Motorcycle** | Deep Sky Blue | `(0, 191, 255)` |
| 🛺 **Three Wheeler** | Yellow | `(255, 255, 0)` |
| 🚜 **Tractor** | Brown | `(165, 42, 42)` |
| � **Truck** | Blue | `(0, 0, 255)` |
| 🚐 **Van** | Cyan | `(0, 255, 255)` |

*Each class is drawn with a unique color for easy visual identification in annotated images.*

## 📈 Detection Output

### Single Image Mode Output:
- Annotated image with color-coded bounding boxes and labels
- Total vehicles detected
- Average confidence score
- Per-class vehicle count
- Horizontal bar chart of class distribution (easier to read)

### Folder Mode Output:
- All annotated images organized by dominant class
- CSV report with per-image statistics
- Overall summary statistics
- Class distribution across all images with horizontal bar chart
- Sample results preview (first 6 images)

## 🔧 Configuration

You can modify detection parameters in the sidebar:

- **Confidence Threshold**: (0.1 - 1.0)
  - Lower values: More detections, potentially more false positives
  - Higher values: Fewer detections, higher precision

## 📝 CSV Report Format

When processing folders, a CSV report is generated with the following columns:
- `Image Name`: Original filename
- `Total Vehicles`: Number of vehicles detected
- `Dominant Class`: Most common vehicle type in the image
- `Average Confidence`: Mean confidence score
- `[Class] Count`: Count for each detected class (e.g., Car Count, Truck Count)

## 🐛 Troubleshooting

### Model Not Found Error
```
⚠️ Please place your YOLOv8 model file (best.pt) in the models/ directory.
```
**Solution**: Copy your trained model to `models/best.pt`

### Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### No Detections
- Try lowering the confidence threshold
- Ensure your images contain vehicles
- Check that the model is trained on similar data

## 🎯 Performance Tips

1. **Image Size**: Larger images take longer to process
2. **Batch Size**: Processing many images simultaneously may use significant memory
3. **Confidence Threshold**: Start with 0.25 and adjust based on results
4. **GPU Support**: YOLOv8 automatically uses GPU if available for faster inference

## 🎨 Customization

### Adding Your Logo
1. Place your logo file as `assets/logo.png`
2. The logo will automatically appear in the sidebar
3. Supported formats: PNG, JPG, JPEG

### Customizing Colors
Edit `style.css` to change the color scheme:
- Navy blues: `#0b3b79`, `#1e5a9e`
- Light blues: `#4a90e2`, `#5ba3f5`
- Accent: `#2c7be5`

### Modifying Fonts
Change font families in `style.css`:
```css
font-family: 'Inter', 'Roboto', sans-serif;
```

For detailed customization guides, see the inline comments in `style.css`.

## 📸 Screenshots

*Add your screenshots here after running the application*

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Single Image Detection
![Single Detection](screenshots/single_detection.png)

### Batch Processing Results
![Batch Results](screenshots/batch_results.png)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available under the MIT License.

## 👏 Acknowledgments

- **YOLOv8** by Ultralytics for the powerful detection model
- **Streamlit** for the amazing web framework
- **OpenCV** for image processing capabilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the quick start guide in `QUICKSTART.txt`
3. Check code comments in `utils/detector.py`, `utils/reporter.py`, and `utils/image_helper.py`
4. Review CSS customization options in `style.css`
5. Ensure all dependencies are correctly installed

## 📁 File Descriptions

- **app.py** (410 lines): Main Streamlit application with UI logic
- **style.css** (441 lines): Complete CSS styling with blue theme
- **utils/detector.py**: YOLOv8 detection logic with 8-class color mapping
- **utils/reporter.py**: CSV report and summary statistics generation
- **utils/image_helper.py**: Logo/icon loading and display functions
- **check_setup.py**: Verify that all components are properly installed
- **test_detector.py**: Unit tests for the detector module

---

**Built with ❤️ using YOLOv8 and Streamlit**

*Last Updated: October 2025*

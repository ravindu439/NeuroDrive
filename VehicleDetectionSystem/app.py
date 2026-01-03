
import streamlit as st
import os
from pathlib import Path
import shutil
from PIL import Image
import pandas as pd
from datetime import datetime
import zipfile

from utils.detector import VehicleDetector
from utils.reporter import generate_report, generate_summary_stats
from utils.image_helper import  display_sidebar_logo
from pathlib import Path
# Page configuration
# Page configuration
st.set_page_config(
    page_title="NeuroDrive",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from external file
def load_css():
    """Load custom CSS from style.css file"""
    css_file = Path("style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(" style.css file not found. Using default styling.")

load_css()



def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "Single Image"


def save_uploaded_file(uploaded_file, upload_dir):
    """Save uploaded file to uploads directory"""
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    initialize_session_state()
    
    # Check if model exists
    model_path = Path("models/best.pt")
    
    if not model_path.exists():
        st.markdown("""
<div class="error-box">
    <h3>Model Not Found</h3>
    <p>Please place your YOLOv8 model file (<b>best.pt</b>) in the <code>models/</code> directory.</p>
    <p>You can copy it from the <code>runs/detect/vehicle_detector3/weights/</code> folder.</p>
</div>
""", unsafe_allow_html=True)
        
        st.code("""
# Copy model from vehicle_detector3
cp ../runs/detect/vehicle_detector3/weights/best.pt models/best.pt
        """, language="bash")
        return

    if Path("assets/logo.png").exists():
        img_base64 = get_img_as_base64("assets/logo.png")
        st.markdown(f"""
        <div class="logo-header">
            <img src="data:image/png;base64,{img_base64}" class="header-logo">
            <div class="logo-text">
                <h1>NeuroDrive</h1>
                <p class="logo-subtitle">Advanced Vehicle Recognition System</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Control Panel
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div class="phase-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.47a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                <h3>Configuration</h3>
            </div>
            """, unsafe_allow_html=True)
            mode = st.radio(
                "Input Mode:",
                ["Single Image", "Folder"],
                key="mode_selector",
                horizontal=True
            )
            
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Minimum confidence score for detections"
            )

        with col2:
            st.markdown("""
            <div class="phase-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                <h3>Upload</h3>
            </div>
            """, unsafe_allow_html=True)
            uploaded_files = None
            if mode == "Single Image":
                uploaded_files = st.file_uploader(
                    "Choose an image file",
                    type=["jpg", "jpeg", "png", "bmp"],
                    accept_multiple_files=False
                )
            else:
                uploaded_files = st.file_uploader(
                    "Choose multiple images",
                    type=["jpg", "jpeg", "png", "bmp"],
                    accept_multiple_files=True
                )

        with col3:
            st.markdown("""
            <div class="phase-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                <h3>Action</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True) # Spacer for alignment
            process_btn = st.button("Classify Vehicles", use_container_width=True, type="primary")
            
            # Mini stats or info could go here
            if uploaded_files:
                count = 1 if mode == "Single Image" else len(uploaded_files)
                st.info(f"Ready to process {count} file(s)")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Main content area
    if process_btn and uploaded_files:
        # Clear previous results
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("uploads"):
            shutil.rmtree("uploads")
        
        os.makedirs("results", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        
        # Initialize detector
        with st.spinner("🔄 Initializing System..."):
            detector = VehicleDetector(str(model_path), confidence_threshold=confidence)
        
        st.success("✅ System Ready!")
        
        # Process based on mode
        if mode == "Single Image":
            # Save uploaded file
            input_path = save_uploaded_file(uploaded_files, "uploads")
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Image")
                st.image(input_path, use_container_width=True)
            
            # Process image
            with st.spinner("Processing..."):
                annotated_path, detections = detector.process_single_image(
                    input_path,
                    "results"
                )
            
            # Display annotated image
            with col2:
                st.subheader("Result")
                st.image(annotated_path, use_container_width=True)
            
            # Display statistics
            st.markdown("---")
            st.subheader("Summary")
            
            # Create statistics cards
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{len(detections['labels'])}</h3>
                    <p>Vehicles Detected</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                avg_conf = sum(detections['confidences']) / len(detections['confidences']) if detections['confidences'] else 0
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{avg_conf:.1%}</h3>
                    <p>Avg Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{len(detections['class_counts'])}</h3>
                    <p>Classes Found</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            if detections['class_counts']:
                st.markdown("### Class Distribution")
                
                class_html = ""
                for class_name, count in sorted(detections['class_counts'].items(), key=lambda x: x[1], reverse=True):
                    class_html += f'<span class="class-badge">{class_name.capitalize()}: {count}</span>'
                
                st.markdown(f'<div style="margin: 1rem 0;">{class_html}</div>', unsafe_allow_html=True)
                
                # Create a horizontal bar chart
                df = pd.DataFrame(list(detections['class_counts'].items()), columns=['Class', 'Count'])
                # Sort by count for better visualization
                df = df.sort_values('Count', ascending=True)
                st.bar_chart(df.set_index('Class'), horizontal=True)
            
            # Download button
            st.markdown("---")
            with open(annotated_path, "rb") as file:
                st.download_button(
                    label="Download Result",
                    data=file,
                    file_name=f"result_{Path(input_path).name}",
                    mime="image/jpeg",
                    use_container_width=True
                )
        
        else:  # Folder mode
            # Save all uploaded files
            st.info(f"Queued {len(uploaded_files)} images...")
            
            folder_path = "uploads/batch"
            os.makedirs(folder_path, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                save_uploaded_file(uploaded_file, folder_path)
            
            # Process folder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing batch..."):
                results = detector.process_folder(
                    folder_path,
                    "results/batch",
                    organize_by_class=True
                )
            
            progress_bar.progress(100)
            status_text.success(f"✅ Processed {len(results)} images")
            
            # Generate summary statistics
            summary = generate_summary_stats(results)
            
            # Display overall statistics
            st.markdown("---")
            st.subheader("Batch Summary")
            
            cols = st.columns(4)
            
            with cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{summary['total_images']}</h3>
                    <p>Images</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{summary['total_vehicles']}</h3>
                    <p>Vehicles</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{summary['avg_confidence']:.1%}</h3>
                    <p>Avg Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{len(summary['class_distribution'])}</h3>
                    <p>Classes</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            st.markdown("### Class Distribution")
            
            if summary['class_distribution']:
                class_html = ""
                for class_name, count in sorted(summary['class_distribution'].items(), key=lambda x: x[1], reverse=True):
                    class_html += f'<span class="class-badge">{class_name.capitalize()}: {count}</span>'
                
                st.markdown(f'<div style="margin: 1rem 0;">{class_html}</div>', unsafe_allow_html=True)
                
                # Create horizontal bar chart
                df = pd.DataFrame(list(summary['class_distribution'].items()), columns=['Class', 'Count'])
                # Sort by count for better visualization
                df = df.sort_values('Count', ascending=True)
                st.bar_chart(df.set_index('Class'), horizontal=True)
            
            # Generate and display report
            st.markdown("---")
            st.subheader("Report")
            
            report_path = "results/report.csv"
            report_df = generate_report(results, report_path)
            
            st.dataframe(report_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download CSV",
                        data=file,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                # Create a zip file of all results
                if os.path.exists("results/batch"):
                    zip_path = "results/batch_results.zip"
                    
                    # Create zip file
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk("results/batch"):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, "results")
                                zipf.write(file_path, arcname)
                    
                    # Provide download button
                    with open(zip_path, "rb") as file:
                        st.download_button(
                            label="Download Results (ZIP)",
                            data=file,
                            file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            
            # Display sample results
            st.markdown("---")
            st.subheader("Samples")
            
            # Show first 6 results
            cols = st.columns(3)
            for idx, result in enumerate(results[:6]):
                with cols[idx % 3]:
                    st.image(result['annotated_path'], caption=result['image_name'], use_container_width=True)
                    st.caption(f"{result['total_vehicles']} vehicles | {result['dominant_class']}")
    
    elif not uploaded_files:
        # Welcome screen
        st.markdown("""
<div class="info-card">
    <h2>Welcome to NeuroDrive</h2>
    <p>Upload your images to begin vehicle detection.</p>
    <h3>Features:</h3>
    <ul>
        <li><b>Single Image:</b> Detect vehicles in one image</li>
        <li><b>Batch Processing:</b> Process multiple images at once</li>
        <li><b>Classification:</b> Automated vehicle categorization</li>
        <li><b>Export:</b> Download reports and images</li>
    </ul>
</div>
""", unsafe_allow_html=True)
        
        # Display model info
        if model_path.exists():
            st.markdown("""
<div class="success-box">
    <h4>System Ready</h4>
    <p>Model loaded: <code>models/best.pt</code></p>
</div>
""", unsafe_allow_html=True)

    # About section at the bottom
    with st.expander("About NeuroDrive"):
        st.markdown("""
        **NeuroDrive** uses **YOLOv8** technology to detect and classify vehicles.
        
        **Supported Classes (8 types):**
        - 🚗 Car
        - 🚚 Truck
        - 🚐 Van
        - 🚌 Bus
        - 🚲 Bicycle
        - 🏍️ Motorcycle
        - 🚜 Tractor
        - 🛺 Three Wheeler
        """)

if __name__ == "__main__":
    main()

"""
Sample Test Script - Vehicle Detection System
This script demonstrates how to use the detector module programmatically
"""

from pathlib import Path
from utils.detector import VehicleDetector
from utils.reporter import generate_report, generate_summary_stats


def test_single_image():
    """Test detection on a single image"""
    print("="*60)
    print("Testing Single Image Detection")
    print("="*60)
    
    # Initialize detector
    model_path = "models/best.pt"
    detector = VehicleDetector(model_path, confidence_threshold=0.25)
    
    # Path to test image (you need to provide this)
    test_image = "../test_images/test1.jpg"  # Update this path
    
    if not Path(test_image).exists():
        print(f" Test image not found: {test_image}")
        print("Please update the path to a valid test image")
        return
    
    # Process image
    print(f"Processing: {test_image}")
    annotated_path, detections = detector.process_single_image(
        test_image,
        "results/test"
    )
    
    # Display results
    print(f"\n Results:")
    print(f"   Annotated image saved: {annotated_path}")
    print(f"   Total vehicles detected: {len(detections['labels'])}")
    print(f"   Classes found: {list(detections['class_counts'].keys())}")
    print(f"\n   Class breakdown:")
    for class_name, count in detections['class_counts'].items():
        print(f"      {class_name}: {count}")
    
    if detections['confidences']:
        avg_conf = sum(detections['confidences']) / len(detections['confidences'])
        print(f"\n   Average confidence: {avg_conf:.2%}")
    
    print("\n" + "="*60)


def test_folder():
    """Test detection on multiple images"""
    print("="*60)
    print("Testing Folder Detection")
    print("="*60)
    
    # Initialize detector
    model_path = "models/best.pt"
    detector = VehicleDetector(model_path, confidence_threshold=0.25)
    
    # Path to test folder (you need to provide this)
    test_folder = "../test_images"  # Update this path
    
    if not Path(test_folder).exists():
        print(f" Test folder not found: {test_folder}")
        print("Please update the path to a valid folder with images")
        return
    
    # Process folder
    print(f"Processing folder: {test_folder}")
    results = detector.process_folder(
        test_folder,
        "results/test_batch",
        organize_by_class=True
    )
    
    # Generate report
    report_df = generate_report(results, "results/test_batch/report.csv")
    summary = generate_summary_stats(results)
    
    # Display results
    print(f"\n Results:")
    print(f"   Images processed: {summary['total_images']}")
    print(f"   Total vehicles detected: {summary['total_vehicles']}")
    print(f"   Average confidence: {summary['avg_confidence']:.2%}")
    print(f"\n   Class distribution:")
    for class_name, count in summary['class_distribution'].items():
        print(f"      {class_name}: {count}")
    
    print(f"\n   Report saved: results/test_batch/report.csv")
    print(f"   Images organized by class in: results/test_batch/")
    
    print("\n" + "="*60)


def main():
    """Main test function"""
    print("\nVehicle Detection System - Test Script\n")
    
    # Check if model exists
    if not Path("models/best.pt").exists():
        print(" Model not found at models/best.pt")
        print("Please copy your model first:")
        print("   cp ../runs/detect/vehicle_detector3/weights/best.pt models/best.pt")
        return
    
    print("Choose test mode:")
    print("1. Test single image detection")
    print("2. Test folder detection")
    print("3. Run both tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_single_image()
    elif choice == "2":
        test_folder()
    elif choice == "3":
        test_single_image()
        print("\n")
        test_folder()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

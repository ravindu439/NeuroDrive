
import pandas as pd
from pathlib import Path


def generate_report(results, output_path):

    # Prepare data for DataFrame
    report_data = []
    
    for result in results:
        row = {
            'Image Name': result['image_name'],
            'Total Vehicles': result['total_vehicles'],
            'Dominant Class': result['dominant_class'],
            'Average Confidence': f"{result['avg_confidence']:.2%}",
        }
        
        # Add per-class counts
        for class_name, count in result['detections']['class_counts'].items():
            row[f'{class_name.capitalize()} Count'] = count
        
        report_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Fill NaN values with 0 for count columns
    count_columns = [col for col in df.columns if 'Count' in col]
    df[count_columns] = df[count_columns].fillna(0).astype(int)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def generate_summary_stats(results):

    if not results:
        return {
            'total_images': 0,
            'total_vehicles': 0,
            'avg_confidence': 0,
            'class_distribution': {}
        }
    
    total_vehicles = sum(r['total_vehicles'] for r in results)
    all_confidences = []
    class_distribution = {}
    
    for result in results:
        all_confidences.extend(result['detections']['confidences'])
        
        for class_name, count in result['detections']['class_counts'].items():
            if class_name in class_distribution:
                class_distribution[class_name] += count
            else:
                class_distribution[class_name] = count
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    return {
        'total_images': len(results),
        'total_vehicles': total_vehicles,
        'avg_confidence': avg_confidence,
        'class_distribution': class_distribution
    }

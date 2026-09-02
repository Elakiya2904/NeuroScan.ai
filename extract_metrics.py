import pdfplumber
import os
import re
import json

pdf_files = [
    './colab/brain_tumor_detection.ipynb - Colab.pdf',
    './colab/segmentation.ipynb - Colab.pdf'
]

metrics_data = {}

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"\n{'='*60}")
        print(f"Extracting from: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                
                # Look for accuracy, precision, recall metrics
                accuracy_pattern = r'accuracy[:\s]+([0-9.]+)'
                precision_pattern = r'precision[:\s]+([0-9.]+)'
                recall_pattern = r'recall[:\s]+([0-9.]+)'
                f1_pattern = r'f1[_-]?score[:\s]+([0-9.]+)'
                loss_pattern = r'loss[:\s]+([0-9.]+)'
                val_acc_pattern = r'val[_-]?acc[uracy]*[:\s]+([0-9.]+)'
                
                # Find all occurrences
                accuracies = re.findall(accuracy_pattern, text, re.IGNORECASE)
                precisions = re.findall(precision_pattern, text, re.IGNORECASE)
                recalls = re.findall(recall_pattern, text, re.IGNORECASE)
                f1_scores = re.findall(f1_pattern, text, re.IGNORECASE)
                losses = re.findall(loss_pattern, text, re.IGNORECASE)
                val_accs = re.findall(val_acc_pattern, text, re.IGNORECASE)
                
                file_name = os.path.basename(pdf_path)
                metrics_data[file_name] = {
                    'accuracies': accuracies[:10],
                    'precisions': precisions[:10],
                    'recalls': recalls[:10],
                    'f1_scores': f1_scores[:10],
                    'losses': losses[:10],
                    'val_accuracies': val_accs[:10]
                }
                
                print(f"Accuracies: {accuracies[:5]}")
                print(f"Precisions: {precisions[:5]}")
                print(f"Recalls: {recalls[:5]}")
                print(f"F1 Scores: {f1_scores[:5]}")
                print(f"Val Accuracies: {val_accs[:5]}")
                print(f"Losses: {losses[:5]}")
                    
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
    else:
        print(f"File not found: {pdf_path}")

# Save to file
with open('extracted_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)

print("\n\nMetrics saved to extracted_metrics.json")

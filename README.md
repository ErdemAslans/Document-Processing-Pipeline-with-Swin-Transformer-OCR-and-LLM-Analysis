# Document Processing System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/transformers-4.30%2B-green)

A comprehensive end-to-end document processing system that combines computer vision, OCR, and NLP for automated document classification, information extraction, and content analysis.

## ✨ Features

- **Document Classification**: Accurate multi-class document classification using Swin Transformer architecture
- **Text Extraction**: OCR capabilities using Unstructured library with support for various formats (PDF, TIF, JPG, PNG)
- **Content Analysis**: LLM-based document understanding using DeepSeek LLM
- **Batch Processing**: Process entire document collections efficiently
- **Interactive UI**: Easy-to-use Gradio interface for visualizing results
- **Optimized Performance**: Support for GPU acceleration and robust error handling

## 📋 Supported Document Types

The system supports 16 document classes:
- Letters
- Forms
- Emails
- Handwritten documents
- Advertisements
- Scientific reports
- Scientific publications
- Specifications
- File folders
- News articles
- Budgets
- Invoices
- Presentations
- Questionnaires
- Resumes
- Memos

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-processing-system.git
cd document-processing-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
unstructured>=0.10.0
pandas>=1.5.0
numpy>=1.24.0
Pillow>=9.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=3.40.0
tqdm>=4.65.0
```

## 🚀 Usage

### Option 1: Using the Interactive UI

```python
from document_processing import DocumentProcessingUI

# Initialize the UI with path to your dataset
ui = DocumentProcessingUI(data_dir="path/to/your/dataset")

# Launch the UI
ui.launch()
```

### Option 2: Programmatic API

```python
from document_processing import DocumentProcessingPipeline

# Initialize the pipeline
pipeline = DocumentProcessingPipeline(data_dir="path/to/your/dataset")

# Process a single document
results = pipeline.process_document("path/to/document.pdf")

# Process a batch of documents
batch_results = pipeline.process_batch(
    folder_path="path/to/documents/",
    max_documents=100,
    output_csv="results.csv"
)

# Train the classifier on your own data
pipeline.train_classifier(
    train_dir="path/to/train",
    val_dir="path/to/val",
    num_epochs=10
)

# Evaluate the classifier
evaluation = pipeline.evaluate_classifier(test_dir="path/to/test")
```

### Google Colab Support

The system includes special support for Google Colab:

```python
# Import the system
from document_processing import start_colab_demo

# Start the demo, which will automatically connect to Google Drive
start_colab_demo()
```

## 📊 Model Architecture

This system uses a Swin Transformer (`microsoft/swin-base-patch4-window7-224-in22k`) fine-tuned for document classification. The architecture includes:

- **Backbone**: Swin Transformer (hierarchical vision transformer)
- **Input Size**: 224x224 images
- **Training**: AdamW optimizer with cosine annealing learning rate schedule
- **Data Augmentation**: Random crops, flips, and color jitter for improved generalization
- **Preprocessing**: Custom TIF file handling and image normalization

## 🗄️ Dataset Structure

The system expects data in the following structure:

```
dataset_root/
├── train/
│   ├── letter/
│   │   ├── image1.tif
│   │   ├── image2.jpg
│   │   └── ...
│   ├── form/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

If you provide a flat structure with class folders, the system will automatically split it into train/val/test sets.

## 🖥️ Interactive UI

The system provides a Gradio-based UI with the following features:

- Upload documents for analysis
- View document classification results with confidence scores
- Read extracted text from documents
- View LLM-based document analysis
- Train and evaluate models through the interface

## 🔍 Document Analysis Capabilities

For each document, the system can:

1. **Classify** the document type with confidence scores
2. **Extract** text content using OCR
3. **Analyze** the document content using LLM to identify:
   - Key entities (people, organizations, dates)
   - Document purpose and structure
   - Important information extraction
   - Document summarization

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- RVL-CDIP dataset for document classification
- Microsoft for the Swin Transformer architecture
- DeepSeek for the LLM model
- Unstructured.io for the document parsing capabilities

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

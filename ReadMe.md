# Car Parking Vacancy Detection System

## Overview
The **Car Parking Vacancy Detection System** is an AI-driven solution for detecting vacant parking spaces using deep learning models. The system is trained on various state-of-the-art object detection architectures, leveraging high-quality annotated datasets for optimal accuracy.

## Models Used
The following deep learning models were utilized to train the system:
- **YOLO (v11)**
- **EfficientDet**
- **RetinaNet**
- **Deformable DETR**
- **Mask R-CNN**
- **Swin Transformer**
- **VGG16**
- **CenterNet**

## Data Annotation
The dataset was annotated using the **Segment Anything Model (SAM)** and **Sreeni** with zero-shot learning techniques. These advanced annotation methods ensured precise labeling of parking spaces, enabling high model performance.

## Dataset Preparation
1. Images of parking areas were collected from multiple sources.
2. Annotation was performed using **SAM** and **Sreeni**.
3. The dataset was split into training, validation, and testing sets.
4. Preprocessing steps such as resizing, normalization, and augmentation were applied.

## Training Process
- The models were trained using **Hugging Face Transformers and PyTorch**.
- A combination of **data augmentation techniques** was applied to improve generalization.
- Models were evaluated using metrics such as **mAP (mean Average Precision), IoU (Intersection over Union), and F1-score**.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- OpenCV
- TensorFlow (if needed for certain models)
- SAM for annotation

### Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/Car-Parking-Vacancy-Detection.git
cd Car-Parking-Vacancy-Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training the Model
```bash
python train.py --model yolov11 --epochs 50 --batch_size 16
```

### Running Inference
```bash
python detect.py --image test_image.jpg --model swin_transformer
```

### Evaluation
```bash
python evaluate.py --model retinaNet
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- Hugging Face for model training frameworks.
- Meta AI for the SAM annotation tool.
- Open-source datasets and community contributions.

## Contact
For inquiries, feel free to reach out via [your-email@example.com](mailto:your-email@example.com) or visit our GitHub repository.


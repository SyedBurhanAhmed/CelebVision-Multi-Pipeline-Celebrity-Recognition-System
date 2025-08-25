
# Model Weights

## 🎯 YOLO Gender Detection Model

- **File:** `yolo_gender_detection.pt` (5.2MB) ✅
- **Architecture:** YOLOv11n fine-tuned for gender detection
- **Classes:** Female (0), Male (1)
- **Performance:** mAP@0.5 = 0.966, Precision > 0.95, Recall > 0.9
- **Training:** 50 epochs on custom annotated dataset (4917 training images)
- **Usage:** Load directly from this repository

```python
from ultralytics import YOLO

# Load YOLO model for face detection and gender classification
yolo_model = YOLO('models/yolo_gender_detection.pt')

# Inference
results = yolo_model('path/to/image.jpg')
```


## 🌟 Celebrity Classifier Model

- **Architecture:** EfficientNetV2-S with custom classifier head
- **Classes:** 15 Pakistani celebrities + "others" class
- **Performance:** F1 = 0.973 (public), 1.000 (private) - **1st place** Kaggle competition
- **Dataset:** Highly imbalanced (2000 "others", 180-200 per celebrity)
- **Size:** ~80MB (hosted on Kaggle for optimal version control)


### Download via KaggleHub:

```python
import kagglehub
import torch
from torchvision.models import efficientnet_v2_s

# Download latest version
path = kagglehub.model_download("syedburhanahmed/cv-project-part-1celebrity-detection/pyTorch/default")
print("Path to model files:", path)

# Load the classifier
model = efficientnet_v2_s(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1280, 512),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 16)  # 15 celebrities + 1 others class
)
model.load_state_dict(torch.load(f"{path}/model.pth"))
model.eval()
```


## 🔄 Complete Pipeline Integration

Both models work together in a two-stage pipeline:

1. **YOLO Detection:** Detects faces and predicts gender (colored bounding boxes)
2. **Celebrity Classification:** Crops detected faces and predicts celebrity identity
```python
# Complete pipeline example
def celebrity_recognition_pipeline(image_path):
    # Stage 1: Face detection + gender classification
    yolo_results = yolo_model(image_path)
    
    # Stage 2: Celebrity classification for each detected face
    for detection in yolo_results:
        cropped_face = crop_face(detection)
        celebrity_prediction = classifier_model(cropped_face)
    
    return combined_results
```


## 🚀 Usage Examples

See these notebooks for complete implementation:

- `notebooks/part1_pakistani_celebrity_classifier.ipynb` - Classifier training \& evaluation
- `notebooks/part2_yolo_gender_detection_training.ipynb` - YOLO training process
- `notebooks/celebvision_complete_pipeline_yolo_classifier.ipynb` - **Full pipeline integration**


## 📊 Performance Summary

| Model | Task | Metric | Score |
| :-- | :-- | :-- | :-- |
| YOLO | Face Detection + Gender | mAP@0.5 | 0.966 |
| YOLO | Gender Classification | Precision | >95% |
| YOLO | Gender Classification | Recall | >90% |
| EfficientNet | Celebrity Recognition | F1 Score | 0.973 |
| EfficientNet | Celebrity Recognition | Validation Accuracy | >98% |

## 🏆 Competition Context

Both models were developed as part of Computer Vision assignments under **Sir Azeem's** guidance, conducted as Kaggle competitions:

- **Part 1:** Celebrity classifier competition - **1st place** out of 50+ students
- **Part 2:** Multi-person gender + celebrity recognition pipeline


## 📋 Requirements

```txt
ultralytics>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
kagglehub>=0.1.0
opencv-python>=4.5.0
pillow>=8.0.0
```


## 🔗 External Links

- 🏆 **Kaggle Classifier Model:** [Pakistani Celebrity Recognition](https://kaggle.com/models/syedburhanahmed/cv-project-part-1celebrity-detection)
- 📊 **Competition Leaderboard:** [CV Project Part 1](https://www.kaggle.com/competitions/s-2025-multi-class-pretraied-network-project)
- 📈 **Training Results:** See `results/` folder for performance visualizations

***

**Note:** The YOLO model is lightweight (5.2KB) and stored directly in this repository, while the larger celebrity classifier is hosted on Kaggle for optimal version control and sharing.

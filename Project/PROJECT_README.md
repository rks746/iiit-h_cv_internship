## Helmet Detection using YOLO

## 1. Introduction

This project is part of the IIIT Hyderabad Computer Vision Internship Program (2025–26).  
The objective is to design a computer vision system that can automatically detect **helmet compliance** among two-wheeler riders using images and videos.  
Helmet detection is a socially relevant problem with applications in road safety enforcement, traffic analytics, and smart surveillance systems. Helmets are highly effective at preventing fatal head injuries (reducing the risk by 42% for motorcyclists and up to 88% for bicyclists).

## 2. Problem Statement

To develop an object detection system that identifies whether a motorcycle rider is wearing a helmet or not, using a deep learning–based computer vision model trained via **transfer learning**.

## 3. Dataset Source and Curation

### 3.1 Original Dataset

The initial dataset was obtained from Kaggle:

> **Kaggle Dataset:**  
> _Bike Riders Helmet Detection Dataset_  
> (link: `https://www.kaggle.com/datasets/abuzarkhaaan/helmet-dataset-cls`)

```
images/
 ├── helmet/
 ├── no helmet/
 └── unlabeled/
```

This dataset was designed for **image classification** and did not contain **bounding box annotations**, making it unsuitable for direct use in object detection tasks.

### 3.2 Dataset Selection

Instead of using the entire dataset, a curated subset was created to ensure:

- visual diversity
- minimal redundancy
- clear visibility of the rider’s head region

From the original dataset, **310 images** with riders with or without helmets were selected and manually annotated.  

Images with poor lighting, occlusion, heavy blur, or unclear head visibility were intentionally excluded.  

### 3.3 Dataset Annotation

To convert the classification dataset into an **object detection dataset**, all selected images were manually annotated using **Label Studio**.  

#### Annotation Tool

- Tool used: **Label Studio**
    
- Installation:
    
    ```bash
    pip install label-studio
    ```
    
#### Annotation Strategy

- Annotation type: **Bounding Box**
    
- Region annotated: **Rider’s head**
    
- Class labels:
    
    - `helmet`
        
    - `nohelmet`
        

Each image contains one or more bounding box corresponding to the rider’s head region, labeled according to helmet presence.  

After annotation, the dataset was exported in **YOLO format**.

## 4. Dataset Structure

The final dataset follows the standard YOLO directory structure:

```
images/
├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── labeled/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## 5. Model and Methodology

### 5.1 Model Selection

The **YOLO11s** model was selected to balance inference efficiency and representational capacity. Compared to lighter variants, YOLO11s provides improved feature extraction while remaining suitable for near real-time deployment, making it appropriate for the scale and complexity of the curated dataset.

The model was initialized using pretrained weights trained on the COCO (Common Object in Context) dataset.

---

### 5.2 Transfer Learning

Transfer learning was performed by fine-tuning the pretrained YOLOv8 model on the custom helmet detection dataset.

Training was executed using the Ultralytics YOLO framework:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

## 6. Evaluation and Analysis

Model performance was evaluated using standard object detection metrics:

- Precision
- Recall
- F1-score
- Confusion Matrix
- Confidence-based analysis

## 7. Conclusion

This project demonstrates the complete pipeline of:

- dataset curation
    
- manual annotation
    
- transfer learning using a pretrained YOLO model
    
- evaluation
    

The results indicate that YOLO-based transfer learning is effective for helmet detection tasks, provided that high-quality annotated data is available.

## 8. References

- Ultralytics YOLOv8 Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com/)
    
- Label Studio Documentation: [https://labelstud.io](https://labelstud.io/)
    
- Kaggle Dataset: https://www.kaggle.com/datasets/abuzarkhaaan/helmet-dataset-cls

---

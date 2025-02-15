# **Automatic Identification and Classification of Diverse Sports using Advanced Deep Learning Models**

## **Overview**
This project focuses on **automatically identifying and classifying diverse sports** using advanced **deep learning models**. It utilizes **image classification techniques** to categorize images into four sports classes:  
- **Cricket**  
- **Tennis**  
- **Badminton**  
- **Swimming**  

The study applies **data preprocessing, augmentation, and deep learning models (DNN, ResNet50, and MobileNetV3)** to classify sports images effectively.

---

## **Dataset**
The dataset is obtained from Kaggle:  
🔗 **[Sports Image Classification Dataset](https://www.kaggle.com/datasets/sidharkal/sports-image-classification/data)**  

### **Dataset Structure**
- **Training Set**: Labeled images of different sports classes  
- **Test Set**: Unlabeled images for classification  

Each image is associated with a unique **image ID** and its corresponding **class label**.

---

## **Project Workflow**
### **1️⃣ Data Preprocessing & Augmentation**
- **Noise Removal**
- **Image Transformations**: Flipping, Rotation, Brightness Adjustment  
- **Normalization & Resizing**  
- **Data Augmentation Techniques**

### **2️⃣ Data Splitting**
- **Training Set**: 70%  
- **Validation Set**: 30%  

### **3️⃣ Model Implementation**
Three deep learning models are trained:
1. **Deep Neural Network (DNN)**
2. **Pretrained ResNet50**
3. **Pretrained MobileNetV3**

### **4️⃣ Model Evaluation**
- **Classification Report** (Precision, Recall, F1-Score)  
- **Confusion Matrix**  
- **Accuracy & Loss Graphs**  

---

## **Technologies Used**
- **Python** 🐍  
- **TensorFlow / Keras**  
- **OpenCV**  
- **Matplotlib / Seaborn**  
- **Scikit-learn**  

---

## **Installation & Usage**
### **Step 1: Clone Repository**
```bash
git clone https://github.com/your-username/sports-classification.git
cd sports-classification
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run Jupyter Notebook**
```bash
jupyter notebook
```
Open the **sports_classification.ipynb** file and run all cells.

---

## **Results & Insights**
📌 **Key Findings:**
- **MobileNetV3** outperformed other models in terms of **accuracy and efficiency**.  
- **Data augmentation** improved generalization and reduced overfitting.  
- **ResNet50** performed well but required more computational resources.  

📊 **Comparison Table:**
| Model       | Accuracy  | Precision | Recall  | F1-Score |
|------------|-----------|-----------|---------|----------|
| **DNN**    | 98%     | 0.98    | 0.98    | 0.98    |
| **ResNet50** | 96%  | 0.93      | 0.92    | 0.92     |
| **MobileNetV3** | 94% | 0.91   | 0.93    | 0.91     |

---

## **Future Improvements**
🚀 **Enhancements:**
- Train with **larger datasets** for better generalization.  
- Implement **hyperparameter tuning** to improve performance.  
- Explore **Transformer-based Vision Models (ViT, Swin Transformer)** for enhanced accuracy.  

---


## **License**
This project is licensed under the **MIT License**.

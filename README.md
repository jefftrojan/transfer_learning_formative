# **Handwritten and Machine-Printed Text Classification Using Transfer Learning**


### Name: Jeffrey Dauda

## mission alignment
This project aligns with the mission of enhancing **education** through technology by streamlining the digitization of handwritten and printed documents. Educational institutions often deal with large volumes of handwritten records, forms, or notes, and classifying these documents can significantly improve administrative efficiency and accessibility. By applying transfer learning, we automate document processing, which helps in making learning materials more accessible and searchable. This supports underserved regions where handwritten materials are still prevalent, contributing to better educational outcomes.


## **Problem Statement**
In this project, we aim to classify images containing both machine-printed and handwritten text into their respective categories. The goal of this project is to build a model that can distinguish between **handwritten text** and **machine-printed text** in document images. The dataset used contains scanned documents with both types of text, and we aim to leverage transfer learning to classify these segments accurately.


### **Dataset**
The dataset used for this task is derived from the IAM Handwriting Forms Dataset. Each image contains a mix of machine-printed and handwritten text, which is manually split into two parts. The images are pre-processed (cropped and resized) to fit the input requirements of pre-trained deep learning models.
- iam-handwritten-forms-dataset [https://paperswithcode.com/dataset/iam]

## **Evaluation Metrics**
To assess the performance of the fine-tuned models, we used the following evaluation metrics:
1. **Accuracy**: The percentage of correctly classified instances out of all instances.
2. **Loss**: A measure of how far the model's predictions are from the actual labels, which helps to guide model optimization.
3. **Precision**: The proportion of true positives out of all positive predictions, indicating how many selected items are relevant.
4. **Recall**: The proportion of true positives out of all actual positives, indicating how many relevant items are selected.
5. **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.

## **Experiments and Findings**
We fine-tuned three different pre-trained models on our dataset: **VGG16**, **ResNet50**, and **InceptionV3**. Each model was initialized with weights pre-trained on the ImageNet dataset and then fine-tuned on the text classification task.

### **Strengths of Transfer Learning**:
- **Improved accuracy with fewer labeled data**: The pre-trained models allowed us to achieve high accuracy even with limited training data, since they were already equipped with useful feature representations from large-scale datasets.
- **Faster convergence**: Fine-tuning pre-trained models led to faster training compared to training from scratch, as the models already had a strong feature extraction foundation.

### **Limitations**:
- **Dataset bias**: The original pre-trained models were trained on ImageNet, which consists of natural images. Fine-tuning such models on domain-specific data (e.g., handwritten/machine-printed text) can introduce some bias if the domain gap is large.
- **Model size**: Pre-trained models such as VGG16 and ResNet50 are large and computationally expensive, which might not be ideal for deployment on resource-constrained environments.

## **Model Evaluation Results**

Below is the table summarizing the performance of the fine-tuned models on the test dataset:

| **Model**   | **Accuracy** | **Loss** | **Precision** | **Recall** | **F1 Score** |
|-------------|--------------|----------|---------------|------------|--------------|
| VGG16       | 85.7%        | 0.53     | 0.86          | 0.84       | 0.85         |
| ResNet50    | 88.2%        | 0.45     | 0.89          | 0.86       | 0.87         |
| InceptionV3 | 90.3%        | 0.39     | 0.91          | 0.90       | 0.90         |

### **Discussion of Results**:
- **InceptionV3** achieved the best performance across most evaluation metrics, with the highest accuracy, precision, recall, and F1 score. Its architecture's ability to capture different scales of features might have contributed to its superior performance.
- **ResNet50** also performed well, benefiting from its deep residual connections that help in learning complex features.
- **VGG16**, while still performing reasonably, lagged behind the other models in terms of accuracy and F1 score, potentially due to its simpler architecture compared to the other two models.

## **Conclusion**
Transfer learning proved to be an effective method for tackling the task of handwritten and machine-printed text classification. The fine-tuned models, especially InceptionV3, provided solid results, but the high computational cost and domain gap remain notable challenges. Future work could explore the use of lighter models and further domain-specific pre-training to improve performance and scalability.

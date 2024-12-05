

# **ANN Classification for Churn Prediction**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/python-v3.8%2B-green)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0%2B-orange)  
![Status](https://img.shields.io/badge/status-Active-brightgreen)

---

## **Project Overview**
This project focuses on predicting customer churn using an Artificial Neural Network (ANN). Churn prediction is critical for businesses to retain customers by understanding their behavior and identifying those likely to leave. The model leverages deep learning to provide accurate predictions based on customer data.

---

## **Key Features**
- **Preprocessing Pipelines**: Includes label encoding, one-hot encoding, and scaling for feature engineering.
- **Model Training and Evaluation**: Deep learning model built using TensorFlow/Keras.
- **Prediction Functionality**: Supports predictions for new customer data.
- **Detailed Experimentation**: Includes Jupyter notebooks for experimentation and model fine-tuning.
- **Logging**: Integrated logging for training and validation.

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries & Frameworks**: 
  - TensorFlow/Keras
  - NumPy
  - Pandas
  - Matplotlib/Seaborn
- **Tools**:
  - Jupyter Notebook
  - Pickle for saving preprocessing objects
  - TensorBoard for visualization

---

## **Folder Structure**
```plaintext
.
├── app.py                    # Main application script
├── Churn_Modelling.csv       # Dataset
├── Experiments.ipynb         # Experimentation notebook
├── prediction.ipynb          # Prediction notebook
├── model.h5                  # Trained model
├── requirements.txt          # Project dependencies
├── logs/                     # Training and validation logs
├── scaler.pkl                # Saved scaler object
├── label_encoder_gender.pkl  # Label encoder for gender
├── onehot_encoder_geo.pkl    # One-hot encoder for geography
└── README.md                 # Project documentation
```

---

## **Installation Guide**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TechWithAkash/ANNClassssification_Churn_Prediction.git
   cd ANNClassssification_Churn_Prediction
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

---

## **Usage**
1. **Data Preparation**: The `Churn_Modelling.csv` dataset is used for training the model. Ensure that the file is in the project root directory.
2. **Training the Model**:
   - Run `Experiments.ipynb` for detailed training and evaluation.
3. **Prediction**:
   - Use `prediction.ipynb` or integrate the `app.py` script for real-time predictions.

---

## **Screenshots**
### **Training Logs in TensorBoard**
![TensorBoard Logs](https://via.placeholder.com/800x400.png?text=TensorBoard+Logs)

### **Churn Predictions**
![Prediction Results](https://via.placeholder.com/800x400.png?text=Prediction+Results)

---

## **Future Improvements**
- Add more features for customer segmentation.
- Integrate additional machine learning models for comparison.
- Deploy the model as a web service using Flask/Django.
- Improve interpretability with SHAP or LIME.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for any proposed changes.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgements**
- The dataset is sourced from **[Kaggle](https://www.kaggle.com/)**.
- TensorFlow/Keras for deep learning.
- Community tutorials and forums for guidance.


# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model('model.h5')

# # Load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# ## streamlit app
# st.title('Customer Churn PRediction')

# # User input
# geography = st.selectbox('Geography', onehot_encodergeo.categories[0])
# gender = st.selectbox('Gender', label_encodergender.classes)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Prepare the input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })

# # One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# # Combine one-hot encoded columns with input data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Predict churn
# prediction = model.predict(input_data_scaled)
# prediction_proba = prediction[0][0]

# st.write(f'Churn Probability: {prediction_proba:.2f}')

# if prediction_proba > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')

# -------UI Updation Code------------------------
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from typing import Tuple, Any

class ChurnPredictionApp:
    """
    Streamlit application for customer churn prediction.
    
    This class encapsulates the entire churn prediction workflow,
    including model loading, data preprocessing, and prediction.
    """
    
    def __init__(self, 
                 model_path: str = 'model.h5', 
                 label_encoder_path: str = 'label_encoder_gender.pkl',
                 onehot_encoder_path: str = 'onehot_encoder_geo.pkl',
                 scaler_path: str = 'scaler.pkl'):
        """
        Initialize the churn prediction application.
        
        Args:
            model_path (str): Path to the saved TensorFlow model
            label_encoder_path (str): Path to the gender label encoder
            onehot_encoder_path (str): Path to the geography one-hot encoder
            scaler_path (str): Path to the feature scaler
        """
        # Load the trained model
        self.model = self._load_model(model_path)
        
        # Load encoders and scaler
        self.label_encoder_gender = self._load_pickle(label_encoder_path)
        self.onehot_encoder_geo = self._load_pickle(onehot_encoder_path)
        self.scaler = self._load_pickle(scaler_path)
    
    @staticmethod
    def _load_model(model_path: str) -> Any:
        """
        Load the TensorFlow model safely.
        
        Args:
            model_path (str): Path to the model file
        
        Returns:
            Loaded TensorFlow model
        """
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise
    
    @staticmethod
    def _load_pickle(file_path: str) -> Any:
        """
        Load a pickle file safely.
        
        Args:
            file_path (str): Path to the pickle file
        
        Returns:
            Loaded object from pickle file
        """
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"Error loading pickle file {file_path}: {e}")
            raise
    
    def _preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data for prediction.
        
        Args:
            input_data (pd.DataFrame): Input customer data
        
        Returns:
            Scaled and preprocessed input data
        """
        # One-hot encode 'Geography'
        geo_encoded = self.onehot_encoder_geo.transform(
            [[input_data['Geography'].values[0]]]
        ).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded, 
            columns=self.onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        
        # Prepare input DataFrame
        processed_data = input_data.drop('Geography', axis=1)
        processed_data['Gender'] = self.label_encoder_gender.transform(
            processed_data['Gender']
        )
        
        # Combine one-hot encoded columns with input data
        final_input = pd.concat([processed_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the input data
        return self.scaler.transform(final_input)
    
    def predict_churn(self, preprocessed_input: np.ndarray) -> Tuple[float, str]:
        """
        Predict customer churn probability.
        
        Args:
            preprocessed_input (np.ndarray): Preprocessed input data
        
        Returns:
            Tuple of churn probability and churn status
        """
        # Predict churn probability
        prediction = self.model.predict(preprocessed_input)
        prediction_proba = float(prediction[0][0])
        
        # Determine churn status
        churn_status = "likely to churn" if prediction_proba > 0.5 else "not likely to churn"
        
        return prediction_proba, churn_status
    
    def render_ui(self):
        """
        Render the Streamlit user interface for churn prediction.
        """
        # Set page configuration
        st.set_page_config(
            page_title="Customer Churn Predictor", 
            page_icon=":bank:", 
            layout="wide"
        )
        
        # Title and description
        st.title("🏦 Customer Churn Prediction")
        st.markdown("""
        ### Predict Customer Churn with Machine Learning
        
        This application uses a trained machine learning model to predict 
        the likelihood of a customer churning based on various features.
        """)
        
        # Create input columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Categorical inputs
            geography = st.selectbox(
                'Geography', 
                self.onehot_encoder_geo.categories_[0]
            )
            gender = st.selectbox(
                'Gender', 
                self.label_encoder_gender.classes_
            )
            has_cr_card = st.selectbox(
                'Has Credit Card', 
                [0, 1], 
                format_func=lambda x: 'Yes' if x == 1 else 'No'
            )
            is_active_member = st.selectbox(
                'Is Active Member', 
                [0, 1], 
                format_func=lambda x: 'Yes' if x == 1 else 'No'
            )
        
        with col2:
            # Numerical inputs
            age = st.slider('Age', 18, 92)
            balance = st.number_input('Balance', min_value=0.0)
            credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
            estimated_salary = st.number_input('Estimated Salary')
            tenure = st.slider('Tenure (Years)', 0, 10)
            num_of_products = st.slider('Number of Products', 1, 4)
        
        # Prediction button
        if st.button('Predict Churn', type='primary'):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Gender': [gender],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_of_products],
                    'HasCrCard': [has_cr_card],
                    'IsActiveMember': [is_active_member],
                    'EstimatedSalary': [estimated_salary],
                    'Geography': [geography]
                })
                
                # Preprocess and predict
                preprocessed_input = self._preprocess_input(input_data)
                prediction_proba, churn_status = self.predict_churn(preprocessed_input)
                
                # Display results
                st.success(f"Churn Probability: {prediction_proba:.2%}")
                st.info(f"The customer is **{churn_status}**.")
                
                # Additional insights
                st.markdown("### Model Prediction Insights")
                st.markdown(f"""
                - **Probability of Churn**: {prediction_proba:.2%}
                - **Churn Status**: {churn_status.capitalize()}
                """)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

def main():
    """
    Main function to run the Streamlit application.
    """
    app = ChurnPredictionApp()
    app.render_ui()

if __name__ == '__main__':
    main()
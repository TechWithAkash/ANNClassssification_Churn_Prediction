# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import pickle
# from typing import Tuple, Any

# class ChurnPredictionApp:
#     """
#     Streamlit application for customer churn prediction.
    
#     This class encapsulates the entire churn prediction workflow,
#     including model loading, data preprocessing, and prediction.
#     """
    
#     def __init__(self, 
#                  model_path: str = 'model.h5', 
#                  label_encoder_path: str = 'label_encoder_gender.pkl',
#                  onehot_encoder_path: str = 'onehot_encoder_geo.pkl',
#                  scaler_path: str = 'scaler.pkl'):
#         """
#         Initialize the churn prediction application.
        
#         Args:
#             model_path (str): Path to the saved TensorFlow model
#             label_encoder_path (str): Path to the gender label encoder
#             onehot_encoder_path (str): Path to the geography one-hot encoder
#             scaler_path (str): Path to the feature scaler
#         """
#         # Load the trained model
#         self.model = self._load_model(model_path)
        
#         # Load encoders and scaler
#         self.label_encoder_gender = self._load_pickle(label_encoder_path)
#         self.onehot_encoder_geo = self._load_pickle(onehot_encoder_path)
#         self.scaler = self._load_pickle(scaler_path)
    
#     @staticmethod
#     def _load_model(model_path: str) -> Any:
#         """
#         Load the TensorFlow model safely.
        
#         Args:
#             model_path (str): Path to the model file
        
#         Returns:
#             Loaded TensorFlow model
#         """
#         try:
#             return tf.keras.models.load_model(model_path)
#         except Exception as e:
#             st.error(f"Error loading model: {e}")
#             raise
    
#     @staticmethod
#     def _load_pickle(file_path: str) -> Any:
#         """
#         Load a pickle file safely.
        
#         Args:
#             file_path (str): Path to the pickle file
        
#         Returns:
#             Loaded object from pickle file
#         """
#         try:
#             with open(file_path, 'rb') as file:
#                 return pickle.load(file)
#         except Exception as e:
#             st.error(f"Error loading pickle file {file_path}: {e}")
#             raise
    
#     def _preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
#         """
#         Preprocess the input data for prediction.
        
#         Args:
#             input_data (pd.DataFrame): Input customer data
        
#         Returns:
#             Scaled and preprocessed input data
#         """
#         # One-hot encode 'Geography'
#         geo_encoded = self.onehot_encoder_geo.transform(
#             [[input_data['Geography'].values[0]]]
#         ).toarray()
#         geo_encoded_df = pd.DataFrame(
#             geo_encoded, 
#             columns=self.onehot_encoder_geo.get_feature_names_out(['Geography'])
#         )
        
#         # Prepare input DataFrame
#         processed_data = input_data.drop('Geography', axis=1)
#         processed_data['Gender'] = self.label_encoder_gender.transform(
#             processed_data['Gender']
#         )
        
#         # Combine one-hot encoded columns with input data
#         final_input = pd.concat([processed_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
#         # Scale the input data
#         return self.scaler.transform(final_input)
    
#     def predict_churn(self, preprocessed_input: np.ndarray) -> Tuple[float, str]:
#         """
#         Predict customer churn probability.
        
#         Args:
#             preprocessed_input (np.ndarray): Preprocessed input data
        
#         Returns:
#             Tuple of churn probability and churn status
#         """
#         # Predict churn probability
#         prediction = self.model.predict(preprocessed_input)
#         prediction_proba = float(prediction[0][0])
        
#         # Determine churn status
#         churn_status = "likely to churn" if prediction_proba > 0.5 else "not likely to churn"
        
#         return prediction_proba, churn_status
    
#     def render_ui(self):
#         """
#         Render the Streamlit user interface for churn prediction.
#         """
#         # Set page configuration
#         st.set_page_config(
#             page_title="Customer Churn Predictor", 
#             page_icon=":bank:", 
#             layout="wide"
#         )
        
#         # Title and description
#         st.title("🏦 Customer Churn Prediction")
#         st.markdown("""
#         ### Predict Customer Churn with Machine Learning
        
#         This application uses a trained machine learning model to predict 
#         the likelihood of a customer churning based on various features.
#         """)
        
#         # Create input columns
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Categorical inputs
#             geography = st.selectbox(
#                 'Geography', 
#                 self.onehot_encoder_geo.categories_[0]
#             )
#             gender = st.selectbox(
#                 'Gender', 
#                 self.label_encoder_gender.classes_
#             )
#             has_cr_card = st.selectbox(
#                 'Has Credit Card', 
#                 [0, 1], 
#                 format_func=lambda x: 'Yes' if x == 1 else 'No'
#             )
#             is_active_member = st.selectbox(
#                 'Is Active Member', 
#                 [0, 1], 
#                 format_func=lambda x: 'Yes' if x == 1 else 'No'
#             )
        
#         with col2:
#             # Numerical inputs
#             age = st.slider('Age', 18, 92)
#             balance = st.number_input('Balance', min_value=0.0)
#             credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
#             estimated_salary = st.number_input('Estimated Salary')
#             tenure = st.slider('Tenure (Years)', 0, 10)
#             num_of_products = st.slider('Number of Products', 1, 4)
        
#         # Prediction button
#         if st.button('Predict Churn', type='primary'):
#             try:
#                 # Prepare input data
#                 input_data = pd.DataFrame({
#                     'CreditScore': [credit_score],
#                     'Gender': [gender],
#                     'Age': [age],
#                     'Tenure': [tenure],
#                     'Balance': [balance],
#                     'NumOfProducts': [num_of_products],
#                     'HasCrCard': [has_cr_card],
#                     'IsActiveMember': [is_active_member],
#                     'EstimatedSalary': [estimated_salary],
#                     'Geography': [geography]
#                 })
                
#                 # Preprocess and predict
#                 preprocessed_input = self._preprocess_input(input_data)
#                 prediction_proba, churn_status = self.predict_churn(preprocessed_input)
                
#                 # Display results
#                 st.success(f"Churn Probability: {prediction_proba:.2%}")
#                 st.info(f"The customer is **{churn_status}**.")
                
#                 # Additional insights
#                 st.markdown("### Model Prediction Insights")
#                 st.markdown(f"""
#                 - **Probability of Churn**: {prediction_proba:.2%}
#                 - **Churn Status**: {churn_status.capitalize()}
#                 """)
                
#             except Exception as e:
#                 st.error(f"An error occurred during prediction: {e}")

# def main():
#     """
#     Main function to run the Streamlit application.
#     """
#     app = ChurnPredictionApp()
#     app.render_ui()

# if __name__ == '__main__':
#     main()
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from typing import Tuple, Any, List, Dict
import plotly.graph_objs as go
import plotly.express as px

class ChurnPredictionApp:
    """
    Advanced Streamlit application for customer churn prediction 
    with enhanced error handling and responsive design.
    """
    
    def __init__(self, 
                 model_path: str = 'model.h5', 
                 label_encoder_path: str = 'label_encoder_gender.pkl',
                 onehot_encoder_path: str = 'onehot_encoder_geo.pkl',
                 scaler_path: str = 'scaler.pkl'):
        """
        Initialize the churn prediction application with robust error handling.
        """
        # Initialize model and preprocessing tools
        self.model = None
        self.label_encoder_gender = None
        self.onehot_encoder_geo = None
        self.scaler = None
        
        # Attempt to load all resources
        self._safe_load_resources(
            model_path, 
            label_encoder_path, 
            onehot_encoder_path, 
            scaler_path
        )
    
    def _safe_load_resources(self, 
                              model_path: str, 
                              label_encoder_path: str, 
                              onehot_encoder_path: str, 
                              scaler_path: str):
        """
        Safely load all required resources with comprehensive error handling.
        """
        try:
            # Load model
            self.model = self._load_model(model_path)
            
            # Load encoders and scaler
            self.label_encoder_gender = self._load_pickle(label_encoder_path)
            self.onehot_encoder_geo = self._load_pickle(onehot_encoder_path)
            self.scaler = self._load_pickle(scaler_path)
            
        except Exception as e:
            st.error(f"Critical Error in Resource Loading: {e}")
            st.error("Please ensure all model files are correctly configured.")
            st.stop()
    
    @staticmethod
    def _load_model(model_path: str) -> Any:
        """
        Safely load the TensorFlow model with enhanced validation.
        """
        try:
            # Validate model path
            if not model_path or not tf.io.gfile.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = tf.keras.models.load_model(model_path)
            
            # Additional model validation
            if model is None:
                raise ValueError("Loaded model is None")
            
            return model
        
        except Exception as e:
            st.error(f"Model Loading Error: {e}")
            raise
    
    @staticmethod
    def _load_pickle(file_path: str) -> Any:
        """
        Safely load pickle files with comprehensive error handling.
        """
        try:
            # Validate file path
            if not file_path or not tf.io.gfile.exists(file_path):
                raise FileNotFoundError(f"Encoder/Scaler file not found: {file_path}")
            
            with open(file_path, 'rb') as file:
                loaded_obj = pickle.load(file)
            
            # Additional validation
            if loaded_obj is None:
                raise ValueError(f"Loaded object from {file_path} is None")
            
            return loaded_obj
        
        except Exception as e:
            st.error(f"Resource Loading Error: {e}")
            raise
    
    def _validate_input(self, input_data: pd.DataFrame) -> bool:
        """
        Validate input data before preprocessing.
        
        Args:
            input_data (pd.DataFrame): Input customer data
        
        Returns:
            bool: True if input is valid, False otherwise
        """
        # Define validation rules
        validation_rules = {
            'CreditScore': (300, 850),
            'Age': (18, 92),
            'Tenure': (0, 10),
            'Balance': (0, None),
            'NumOfProducts': (1, 4),
            'EstimatedSalary': (0, None)
        }
        
        # Check each numerical feature
        for feature, (min_val, max_val) in validation_rules.items():
            value = input_data[feature].values[0]
            
            # Skip None max_val checks
            if max_val is not None and (value < min_val or value > max_val):
                st.error(f"Invalid {feature}: Must be between {min_val} and {max_val}")
                return False
            
            # Check for minimum value
            if min_val is not None and value < min_val:
                st.error(f"Invalid {feature}: Must be at least {min_val}")
                return False
        
        return True
    
    def _preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Advanced input preprocessing with comprehensive error handling.
        """
        try:
            # Validate input first
            if not self._validate_input(input_data):
                st.stop()
            
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
            scaled_input = self.scaler.transform(final_input)
            
            return scaled_input
        
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()
    
    def predict_churn(self, preprocessed_input: np.ndarray) -> Tuple[float, str]:
        """
        Advanced churn prediction with detailed probability interpretation.
        """
        try:
            # Predict churn probability with error handling
            if preprocessed_input is None or len(preprocessed_input) == 0:
                st.error("Invalid input for prediction")
                st.stop()
            
            prediction = self.model.predict(preprocessed_input)
            
            if len(prediction) == 0:
                st.error("Model returned empty prediction")
                st.stop()
            
            prediction_proba = float(prediction[0][0])
            
            # Sophisticated churn status determination
            if prediction_proba < 0.3:
                churn_status = "Low Risk"
            elif prediction_proba < 0.6:
                churn_status = "Moderate Risk"
            else:
                churn_status = "High Risk"
            
            return prediction_proba, churn_status
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()
    
    def _create_churn_gauge(self, probability: float) -> go.Figure:
        """
        Create an interactive Plotly gauge chart for churn probability.
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(autosize=True)
        return fig
    
    def render_ui(self):
        """
        Responsive Streamlit user interface with advanced design.
        """
        # Advanced page configuration
        st.set_page_config(
            page_title="Churn Prediction Intelligence", 
            page_icon=":chart_with_upwards_trend:", 
            layout="wide"
        )
        
        # Custom CSS for responsiveness and modern design
        st.markdown("""
        <style>
        /* Responsive design */
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
            }
        }
        
        /* Modern, clean UI */
        .stApp {
            background-color: #f4f6f9;
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2563eb;
            transform: scale(1.05);
        }
        .header {
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stMarkdown h1 {
            color: #1e3a8a;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Professional header
        st.markdown("""
        <div class="header">
            <h1>🏦 Customer Churn Prediction Intelligence</h1>
            <p>Advanced Machine Learning Insights for Customer Retention</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Responsive input columns
        col1, col2 = st.columns([1, 1], gap="medium")
        
        # Default values to prevent potential errors
        default_inputs = {
            'geography': 'France',
            'gender': 'Male',
            'has_cr_card': 1,
            'is_active_member': 1,
            'age': 35,
            'balance': 50000.0,
            'credit_score': 650,
            'estimated_salary': 75000.0,
            'tenure': 5,
            'num_of_products': 2
        }
        
        with col1:
            st.markdown("#### 📊 Customer Profile")
            geography = st.selectbox(
                'Geographic Region', 
                self.onehot_encoder_geo.categories_[0],
                index=self.onehot_encoder_geo.categories_[0].tolist().index(default_inputs['geography']),
                help="Select the customer's geographic location"
            )
            gender = st.selectbox(
                'Gender', 
                self.label_encoder_gender.classes_,
                index=list(self.label_encoder_gender.classes_).index(default_inputs['gender']),
                help="Select the customer's gender"
            )
            has_cr_card = st.selectbox(
                'Credit Card Ownership', 
                [0, 1], 
                index=default_inputs['has_cr_card'],
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help="Does the customer have a credit card?"
            )
            is_active_member = st.selectbox(
                'Membership Status', 
                [0, 1], 
                index=default_inputs['is_active_member'],
                format_func=lambda x: 'Active' if x == 1 else 'Inactive',
                help="Is the customer an active member?"
            )
        
        with col2:
            st.markdown("#### 🔍 Detailed Metrics")
            age = st.slider(
                'Age', 
                min_value=18, 
                max_value=92, 
                value=default_inputs['age'],
                help="Customer's age"
            )
            balance = st.number_input(
                'Account Balance', 
                min_value=0.0, 
                value=default_inputs['balance'],
                help="Customer's account balance"
            )
            credit_score = st.number_input(
                'Credit Score', 
                min_value=300, 
                max_value=850, 
                value=default_inputs['credit_score'],
                help="Customer's credit score"
            )
            estimated_salary = st.number_input(
                'Estimated Salary', 
                min_value=0.0, 
                value=default_inputs['estimated_salary'],
                help="Customer's estimated annual salary"
            )
            tenure = st.slider(
                'Years with Bank', 
                min_value=0, 
                max_value=10, 
                value=default_inputs['tenure'],
                help="Duration of customer's relationship with the bank"
            )
            num_of_products = st.slider(
                'Number of Bank Products', 
                min_value=1, 
                max_value=4, 
                value=default_inputs['num_of_products'],
                help="Number of bank products customer uses"
            )
        
        # Prediction button with enhanced styling
        prediction_btn = st.button('Generate Churn Prediction', type='primary')
        
        if prediction_btn:
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
                
                # Results display
                st.markdown("## 🎯 Prediction Results")
                
                # Responsive results layout
                col_gauge, col_insights = st.columns([1, 1])
                
                with col_gauge:
                    st.plotly_chart(self._create_churn_gauge(prediction_proba), use_container_width=True)
                
                with col_insights:
                    st.metric("Churn Risk Level", churn_status)
                    
                    # Color-coded risk description
                    if churn_status == "Low Risk":
                        st.success("Customer appears stable and likely to continue engagement.")
                    elif churn_status == "Moderate Risk":
                        st.warning("Customer shows signs of potential churn. Proactive retention strategies recommended.")
                    else:
                        st.error("High risk of customer churning. Immediate intervention required.")
                
                # Detailed insights section
                with st.expander("Detailed Churn Insights"):
                    st.markdown(f"""
                    ### Comprehensive Churn Analysis
                    
                    **Probability Breakdown:**
                    - Churn Probability: **{prediction_proba:.2%}**
                    - Risk Classification: **{churn_status}**
                    
                    **Potential Retention Strategies:**
                    - Review customer's product portfolio
                    - Personalized engagement communication
                    - Targeted retention offers
                    """)
                
            except Exception as e:
                st.error(f"Prediction Process Error: {e}")

def main():
    """
    Entry point for the Churn Prediction Streamlit Application.
    """
    app = ChurnPredictionApp()
    app.render_ui()

if __name__ == '__main__':
    main()
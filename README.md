# phishing-url-dectection
“A hybrid machine learning model to detect phishing URLs using Random Forest and Logistic Regression.”
1. Machine Learning (/src)
This is where the core logic of phishing detection is implemented.
Goal: Build and use a model to classify URLs
Work:
Extracted features from URLs (HTTPS, length, special characters)
Applied Machine Learning model (Scikit-learn)
Performed prediction based on input URL
2. Application UI (/src)
This provides the interface to interact with the system.
Goal: Allow users to check URLs easily
Work:
Built UI using Streamlit
Took user input (URL)
Displayed prediction results
3. Prediction Flow
This connects all parts of the system.
Flow:
User Input → Feature Extraction → ML Model → Result
Output:
 Phishing URL
 Safe URL
 Quick Start
Terminal:
cd src
pip install -r requirements.txt
streamlit run app.py

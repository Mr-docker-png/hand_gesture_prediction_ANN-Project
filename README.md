Hand Gesture Classification using ANN

A deep learning project that uses a feed-forward Artificial Neural Network (ANN) to classify static hand gestures from 3D hand-landmark coordinates.
Problem Statement
The goal is to classify a hand into one of four gesture classes:
- close_palm
- open_palm
- peace
- thumbs_up
Instead of using raw images, the model receives 21 hand landmarks, each represented by (x, y, z) coordinates.
21 landmarks × 3 coordinates = 63 input features

Dataset
The dataset contains:
- 798 samples
- 63 numerical features
- 4 gesture classes
- No missing values
- No duplicate rows
Feature structure:
x0 ... x20
y0 ... y20
z0 ... z20

Target:
label

Class distribution
Gesture	Samples
thumbs_up	438
open_palm	120
close_palm	120
peace	120


The dataset is therefore imbalanced, with thumbs_up having significantly more samples.
Project Pipeline
Hand Landmark Dataset
        ↓
Data Exploration
        ↓
Train/Test Split
        ↓
Label Encoding
        ↓
Feature Scaling
        ↓
Baseline ANN
        ↓
Evaluation
        ↓
Error Analysis
        ↓
Dropout Experiment
        ↓
Class-Weighted ANN
        ↓
Final Evaluation
        ↓
Prediction Function

Preprocessing
Train/Test Split
The data was split using a stratified train/test split so that the class distribution was preserved.
Label Encoding
The string labels were converted into integer classes:
close_palm → 0
open_palm  → 1
peace      → 2
thumbs_up  → 3

Feature Scaling
StandardScaler was applied to the 63 input features.
The scaler was fitted only on the training data and then used to transform the test data.
Models
1. Baseline ANN
Architecture:
Input: 63
   ↓
Dense(64, ReLU)
   ↓
Dense(32, ReLU)
   ↓
Dense(4, Softmax)

The model was trained using:
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Metric: Accuracy
- Batch size: 32
Result
Test Accuracy: 98%
2. ANN with Dropout
Architecture:
Input: 63
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.3)
   ↓
Dense(64, ReLU)
   ↓
Dropout(0.3)
   ↓
Dense(4, Softmax)

Result
Test Accuracy: 95%
Dropout did not improve this project.
This suggests that the baseline model was already generalizing well, so additional regularization was unnecessary.
3. Class-Weighted ANN — Final Model
The same baseline architecture was trained using class weights to compensate for the imbalanced dataset.
Input: 63
   ↓
Dense(64, ReLU)
   ↓
Dense(32, ReLU)
   ↓
Dense(4, Softmax)

Class weights were calculated using:
compute_class_weight(    class_weight="balanced",    classes=classes,    y=y_train)


Result
Test Accuracy: 99.5%
Model Comparison
Model	Test Accuracy
Baseline ANN	98.0%
ANN + Dropout	95.0%
Class-Weighted ANN	99.5% 🏆


The class-weighted ANN performed best.
Final Evaluation
Classification report for the final model:
Class	Precision	Recall	F1
close_palm	1.00	0.97	0.98
open_palm	1.00	1.00	1.00
peace	1.00	1.00	1.00
thumbs_up	0.99	1.00	1.00


Overall:
Accuracy: 99.5%
The final model correctly classified 199 out of 200 test samples.
Error Analysis
The baseline model made four errors:
open_palm → peace
close_palm → thumbs_up
close_palm → thumbs_up
thumbs_up → close_palm

After applying class weighting, only one test sample was misclassified:
close_palm → thumbs_up

The final confusion matrix showed:
                 Predicted
              close open peace thumbs
Actual close     29    0    0     1
       open       0   30    0     0
       peace      0    0   30     0
       thumbs     0    0    0   110

Training Analysis
The final model's training and validation curves showed:
- Training accuracy approaching 99–100%
- Validation accuracy approaching 99%
- Training and validation loss decreasing
- No major generalization gap
- No obvious severe overfitting
Therefore, additional regularization was not necessary.
Prediction
A prediction function was created that accepts the 63 landmark coordinates and returns the predicted gesture and confidence.
Example:
Prediction: thumbs_up
Confidence: 80.9%

The complete prediction pipeline is:
Raw landmarks
      ↓
StandardScaler
      ↓
ANN
      ↓
Softmax probabilities
      ↓
Predicted class
      ↓
Original gesture label

Saved Model Files
The trained model and preprocessing objects were saved:
hand_gesture_ann.keras
scaler.pkl
label_encoder.pkl

These allow the trained model to be reused without retraining.
Key Learnings
1. A simple ANN can work very well with engineered features
The model does not need raw images because the hand landmarks already provide meaningful geometric information.
2. Class imbalance matters
The dataset contained many more thumbs_up samples than the other gestures.
Using class weights improved the test accuracy from:
98.0% → 99.5%

3. More regularization is not always better
Adding Dropout reduced performance:
98.0% → 95.0%

This was a useful experiment because it showed that model improvements should be validated experimentally, not assumed.
4. Accuracy is not enough
We also examined:
- Precision
- Recall
- F1-score
- Confusion matrix
- Individual prediction confidence
Future Improvements
Possible next steps:
- Real-time webcam hand gesture recognition
- Extract hand landmarks directly using MediaPipe
- Build a live prediction pipeline
- Add more gesture classes
- Collect more diverse hand poses
- Test robustness to different hand orientations
- Compare ANN with a CNN using raw hand images
- Deploy the model as a small real-time application
Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
Final Conclusion
A feed-forward ANN was successfully developed to classify four hand gestures from 3D hand-landmark coordinates.
The class-weighted ANN achieved 99.5% test accuracy, outperforming both the baseline ANN and the Dropout-based model.
The project demonstrates that careful preprocessing, evaluation, error analysis, and handling class imbalance can be more valuable than simply making a neural network larger or more complex.

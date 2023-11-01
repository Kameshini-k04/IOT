#IOT Phase wise project Submission

# IOT Based Noise Pollution Monitoring Machine Learning

Dataset Link: https://chat.openai.com/share/9304b28c-cdb9-4bbc-8214-1b05c5641e30
Reference:chatGPT.com(openai)
There is no link given in skillup,so we use openai for data analysis.

#how to run the code and dependency

Noise pollution monitoringusing Machine Learning
#How to run:

install jupitor notebook in command promp!

#pip install jupitor lab
# pip install Jupiler notebook (or)
Download Anaconda community software for desklop
install the anaconda community
open Jupiler nolebook
type the code & execute the given code
## Noise pollution monitoring Using Machine Learning

# This project demonstrates noise pollution monitoring machine learning techniques.
## Installation

Explain how to install or set up a project.Include Installation steps such as,
clone the repository:https://github.com/Kameshini-k04/IOT.git
pip install -requirements.txt
# Import necessary libraries

Import numpy as np

From sklearn.model_selection import train_test_split

From sklearn.ensemble import RandomForestClassifier

From sklearn.metrics import accuracy_score

# Load your dataset and extract features (e.g., using librosa library)

# X contains features, y contains corresponding labels (noise levels)

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier (you can choose other classifiers too)

Classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model

Classifier.fit(X_train, y_train)

# Make predictions on the test set

Predictions = classifier.predict(X_test)

# Evaluate the model

Accuracy = accuracy_score(y_test, predictions)

Print(f”Accuracy: {accuracy}”)
Creating a real-time noise level monitoring system with machine learning involves continuous data acquisition and model updates.

Import numpy as np

From sklearn.ensemble import RandomForestRegressor

Import sounddevice as sd  # for real-time audio input

Import librosa  # for audio feature extraction



# Initialize model and variables

Model = RandomForestRegressor(n_estimators=100, random_state=42)

Previous_prediction = None



# Define a callback function for real-time audio processing

Def callback(indata, frames, time, status):

    If status:

        Print(status)

    # Extract audio features from the incoming real-time data

    Features = extract_features(indata)



    # Make prediction using the trained model

    Current_prediction = model.predict(features.reshape(1, -1))[0]



    # Update or notify based on the prediction

    Update_system(current_prediction)



# Function to extract features from audio data

Def extract_features(audio_data):

    # Use librosa or other libraries to extract relevant features

    # Example: mfccs = librosa.feature.mfcc(audio_data.flatten(), sr=sample_rate)

    # Return a 1D array of features



# Function to update or notify based on the prediction

Def update_system(prediction):

    Global previous_prediction

    # Implement logic to update or notify based on the current and previous predictions

    # Example: If noise level increases significantly, send an alert



    # Update the previous prediction

    Previous_prediction = prediction



# Open a stream for real-time audio input

With sd.InputStream(callback=callback):

    Print(‘Real-time noise monitoring started. Press Ctrl+C to stop.’)

    Sd.sleep(duration=np.inf)
Before you begin, ensure you have the following installed:

Python (version 3.6 or higher) – Download Python
Pip – Python package installer, usually included with Python installation.

## Installation
Clone the repository to your local machine:

bashCopy code
git clone:https://github.com/Kameshini-k04/IOT.git

## Navigate to the project directory:

bashCopy code
cd diabetes-prediction

Install the required packages using pip:

bashCopy code
pip install -r requirements.txt

## Usage
Prepare the Dataset:

Download the dataset from the UCI Machine Learning Repository.
Save the dataset as noise pollution monitoring_data.csv in the project directory.
Run the Python Script:

Execute the Python script to train the model and make predictions:
bashCopy code
python noise pollution.py

## View Results:

The script will output the accuracy of the model and a classification report detailing.
Contributing
Feel free to contribute to this project. Fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License – see the LICENSE file for details.






# Emotion-Detection-Machine-Learning-Model-Refinement
Developed a Flask web app using a custom-trained CNN in Keras to classify 7 emotion classes from facial images. Built an image preprocessing pipeline, Flask API for uploads and predictions, and a simple HTML frontend for instant emotion analysis with labels and confidence scores.

# Goal:
The client wanted an intuitive web-based interface to analyze facial emotion from uploaded images, providing real-time predictions of emotional states such as happy, sad, angry, and others, to support emotion recognition tasks.

# Solution:
I developed a Flask web application that loads a custom-trained Convolutional Neural Network (CNN) model to classify emotions from facial images. The solution included:

Training a CNN on a labeled facial expression dataset using Keras, achieving high accuracy for 7 emotion classes.

Creating an image preprocessing pipeline that resizes and normalizes input images to match the model requirements.

Building a Flask backend API that accepts image uploads, runs predictions, and returns JSON with predicted labels and confidence scores.

Designing a simple and user-friendly HTML frontend for users to upload images and receive instant emotion analysis results.

# Impact:
The project enabled rapid, accurate emotion classification via a web interface, making it accessible for research, user experience studies, and human-computer interaction applications. This end-to-end solution simplified the emotion recognition workflow and delivered actionable insights from images within seconds.

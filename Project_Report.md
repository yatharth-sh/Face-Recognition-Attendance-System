# Smart Attendance System with Facial Recognition

## Abstract

Face recognition technology has evolved into various sectors such as education, biometric authentication, and security, making it extremely efficient in tracing attendance and eliminating the need for physical ID cards. This project implements a web-based attendance system using Flask and OpenCV, offering a contactless and automated solution for modern institutions.

## 1. Introduction

### 1.1 Background

Facial recognition technology verifies identity by analyzing and comparing facial features from digital images. Unlike traditional desktop-only applications, this project leverages web technologies to create an accessible interface that runs in a browser, powered by modern, lightweight AI models (YuNet and SFace) for real-time performance.

### 1.2 Problem Statement

Manual attendance tracking is time-consuming, prone to errors, and inefficient. Physical ID cards can be lost or forgotten. This project eliminates these drawbacks by offering a non-intrusive, contactless solution. When a registered face is detected, the system automatically logs the student's name and timestamp into a secure record file.

### 1.3 Objectives

The primary objectives are:

- To efficiently track attendance in real-time using a webcam.
- To automate the logging of "In" and "Out" times in a CSV file.
- To modernize attendance tracking by combining web accessibility with robust computer vision.

## 2. Methodology

### 2.1 Tools and Technologies Used

This project is built using the following key technologies:

1. **Python**: The core programming language used for backend logic and AI processing.
2. **Flask**: A lightweight web framework used to create the user interface and handle video streaming to the browser.
3. **OpenCV**:
    - **YuNet**: A high-performance Convolutional Neural Network (CNN) used for face detection.
    - **SFace**: A state-of-the-art model used for face recognition (feature extraction).
4. **Scikit-learn (KNN)**: The K-Nearest Neighbors algorithm is used to classify face embeddings and identify users based on the trained dataset.
5. **Pandas**: A data manipulation library used to manage and export attendance records in CSV format.

### 2.2 Project Design

The data flow for the project is as follows:

**Start** -> **Image Acquisition** (Webcam feed via Flask) -> **Face Detection** (YuNet locates faces) -> **Feature Extraction** (SFace generates 128-D embeddings) -> **Face Recognition** (KNN Classifier matches embedding to database) -> **Attendance Logging** (Pandas updates CSV) -> **End**.

> [!NOTE]
> *[Insert Diagram: System Architecture Flowchart here]*

### 2.3 Implementation Details

The system is divided into two main components: the **Face System** (backend logic) and the **Web Application** (interface).

**Key Functions:**

1. **`register_face(image, name)`**: Captures a face, extracts its embedding, and stores it in the dataset (`face_data.pkl`).
2. **`train_classifier()`**: Trains the KNN classifier on the collected embeddings to enable recognition.
3. **`recognize(image)`**: Detects faces in a frame, calculates their embeddings, and predicts the identity using the trained classifier.
4. **`log_attendance(name, mode)`**: Records the timestamp for "IN" or "OUT" events in `attendance_log.csv`.

## 3. Results and Discussion

### 3.1 Project Outcomes

The system successfully operates in a web browser, allowing users to:

- Register new faces via a user-friendly interface.
- View a live video feed with real-time bounding boxes and name labels.
- Automatically mark attendance without manual intervention.
- View and export attendance records as a CSV file.

> [!NOTE]
> *[Insert Screenshot: Dashboard with Live Video Feed]*

> [!NOTE]
> *[Insert Screenshot: Attendance Records Table]*

### 3.2 Challenges Faced

1. **Lighting Conditions**: Extreme lighting (too dark or too bright) can affect detection accuracy.
2. **Threshold Tuning**: Finding the optimal distance threshold (currently set to 0.4) for SFace was critical to balance between recognizing valid users and rejecting unknown faces.
3. **Browser Integration**: Streaming video efficiently from Python to a web browser required implementing a specific multipart response format.

### 3.3 Learnings and Insights

Developing this project provided deep insights into:

- **AI Integration**: How to embed complex AI models into a standard web application.
- **Data Persistence**: Managing binary data (pickle files) for models and structured data (CSV) for logs.
- **Real-time Processing**: Optimizing code to ensure the video feed remains smooth while performing heavy calculations.

## 4. Conclusion

This project demonstrates a functional, efficient, and modern approach to attendance tracking. By combining the accessibility of a web interface with the power of OpenCV's latest models, it offers a significant upgrade over manual methods.

**Future Scope:**

- **Cloud Integration**: Storing data centrally for multi-device access.
- **Mobile App**: Developing a native app for easier access.

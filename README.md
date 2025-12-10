# Smart Attendance System with Facial Recognition

A modern, web-based attendance system powered by **Flask** and **OpenCV**. This project uses state-of-the-art AI models (YuNet and SFace) to provide contactless, real-time attendance tracking.

---

## 📋 Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Future Scope](#-future-scope)

---

## 📖 About

Face recognition technology has evolved into a vital tool for security and automation. This project implements a **Smart Attendance System** that eliminates the need for physical ID cards or manual roll calls.

By leveraging **YuNet** for face detection and **SFace** for feature extraction, the system offers a robust and lightweight solution that runs directly in your web browser.

---

## ✨ Key Features

- **Real-Time Recognition**: Instantly detects and identifies registered users from a live video feed.
- **Web-Based Interface**: Accessible via any modern web browser; no desktop app installation required.
- **Automated Logging**: Automatically records "IN" and "OUT" timestamps in a CSV file (`attendance_log.csv`).
- **User Management**: Easy-to-use interface for registering new faces and training the model.
- **Data Export**: View and download attendance records directly from the dashboard.

---

## 🛠 Tech Stack

- **Language**: Python 3.x
- **Web Framework**: Flask
- **Computer Vision**: OpenCV (YuNet, SFace)
- **Machine Learning**: Scikit-learn (K-Nearest Neighbors)
- **Data Handling**: Pandas, NumPy

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher installed on your system.
- A webcam connected to your computer.

### Quick Start (Windows)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Face-Recognition-Attendance-System.git
   cd Face-Recognition-Attendance-System
   ```

2. **Run the Application**
   Simply double-click the `run.bat` file.

   *This script will automatically:*
   - Check for Python.
   - Install all required dependencies (`requirements.txt`).
   - Start the Flask server.

3. **Access the Dashboard**
   Open your browser and navigate to:
   `http://localhost:5000`

### Manual Installation

If you prefer to run it manually:

```bash
pip install -r requirements.txt
python app.py
```

---

## 📂 Project Structure

```
Face-Recognition-Attendance-System/
├── app.py          # Main Flask Application
├── face_system.py          # Core Face Recognition Logic
├── requirements.txt        # Project Dependencies
├── run.bat                 # Auto-run Script for Windows
├── templates/              # HTML Templates (Dashboard, Login, etc.)
├── attendance_log.csv      # Auto-generated Attendance Records
├── face_classifier.pkl     # Trained KNN Model
├── face_data.pkl           # Stored Face Embeddings
└── models/                 # ONNX Models (Downloaded automatically)
```

---

## 🔮 Future Scope

- **Cloud Integration**: Centralized database for multi-location access.
- **Mobile Application**: Native Android/iOS app for easier mobile usage.
- **Advanced Analytics**: Visual reports and graphs for attendance trends.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

import cv2
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, session
from face_system import FaceSystem
import time
import threading
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo'  # Change this in production

# Initialize Face System
face_sys = FaceSystem()

# Global state
app_state = {
    "collection_active": False,
    "collection_name": "",
    "collection_count": 0,
    "collection_target": 20,
    "session_status": "IDLE",  # IDLE, IN_WINDOW, OUT_WINDOW
    "window_end_time": 0
}

def generate_frames():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cam.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        
        # 1. Data Collection Mode
        if app_state["collection_active"]:
            cv2.putText(frame, f"COLLECTING: {app_state['collection_count']}/{app_state['collection_target']}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if app_state["collection_count"] < app_state["collection_target"]:
                success, msg = face_sys.register_face(frame, app_state["collection_name"])
                if success:
                    app_state["collection_count"] += 1
                else:
                    cv2.putText(frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                app_state["collection_active"] = False
                app_state["collection_name"] = ""
                # Auto-train after collection
                face_sys.train_classifier()
                
        # 2. Recognition Mode
        else:
            results = face_sys.recognize(frame)
            
            # Check if window is open
            current_time = time.time()
            window_open = False
            if app_state["session_status"] != "IDLE":
                if current_time < app_state["window_end_time"]:
                    window_open = True
                else:
                    app_state["session_status"] = "IDLE" # Window expired

            for res in results:
                box = res['box']
                name = res['name']
                conf = res['confidence']
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw Box
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                
                # Draw Label Background
                label = f"{name} ({int(conf*100)}%)"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (box[0], box[1]-20), (box[0]+w, box[1]), color, -1) # Filled
                
                # Draw Text
                text_color = (255, 255, 255) if name != "Unknown" else (255, 255, 255)
                cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Log Attendance if window is open and person is known
                if window_open and name != "Unknown":
                    mode = "IN" if app_state["session_status"] == "IN_WINDOW" else "OUT"
                    face_sys.log_attendance(name, mode)

            # Draw Status on Screen
            if window_open:
                status_text = f"WINDOW OPEN ({app_state['session_status']})"
                timer = int(app_state["window_end_time"] - current_time)
                cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Time Left: {timer}s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Routes ---

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            session['user'] = 'admin'
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/records')
def records():
    if 'user' not in session: return redirect(url_for('login'))
    
    csv_file = "attendance_log.csv"
    records = []
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            df = pd.read_csv(csv_file)
            # Sort by Date descending
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by='Date', ascending=False)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            records = df.to_dict(orient='records')
        except pd.errors.EmptyDataError:
            records = []
    return render_template('records.html', records=records)

@app.route('/video_feed')
def video_feed():
    if 'user' not in session: return Response("Unauthorized", 401)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Endpoints ---

@app.route('/api/status')
def get_status():
    remaining = max(0, app_state["window_end_time"] - time.time())
    return jsonify({
        "status": app_state["session_status"],
        "remaining": remaining
    })

@app.route('/api/session/<type>', methods=['POST'])
def start_session(type):
    if type not in ['IN', 'OUT']: return jsonify({"error": "Invalid type"}), 400
    
    duration = 10 * 60 # 10 minutes
    app_state["session_status"] = f"{type}_WINDOW"
    app_state["window_end_time"] = time.time() + duration
    
    return jsonify({"message": f"Started {type} window for 10 minutes."})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    if not name: return jsonify({"message": "Name required"}), 400
    
    app_state["collection_name"] = name
    app_state["collection_count"] = 0
    app_state["collection_active"] = True
    
    return jsonify({"message": f"Started collecting data for {name}..."})

@app.route('/api/train', methods=['POST'])
def train():
    success, msg = face_sys.train_classifier()
    return jsonify({"message": msg})

@app.route('/api/export_csv')
def export_csv():
    if 'user' not in session: return redirect(url_for('login'))
    csv_file = "attendance_log.csv"
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            csv_content = f.read()
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=attendance_log.csv"}
        )
    return "No records found."

@app.route('/api/reset', methods=['POST'])
def reset_data():
    if 'user' not in session: return jsonify({"message": "Unauthorized"}), 401
    success, msg = face_sys.reset_data()
    return jsonify({"message": msg})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
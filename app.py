# from flask import Flask, render_template, Response, request, jsonify
# from detection import detect_and_classify
# import cv2
# import mysql.connector
# from datetime import datetime

# app = Flask(__name__)

# camera = cv2.VideoCapture(0)  # Use 0 for USB Camera

# # MySQL Database configuration
# db_config = {
#     'user': 'root',
#     'password': '134500',
#     # 'host': '127.0.0.1',
#     'host':'localhost',
#     'database': 'ranking_db'
# }

# def get_db_connection():
#     conn = mysql.connector.connect(**db_config)
#     return conn

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     def gen_frames():
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             else:
#                 # Perform detection and classification
#                 processed_frame, sections = detect_and_classify(frame)

#                 ret, buffer = cv2.imencode('.jpg', processed_frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/detect', methods=['POST'])
# def detect():
#     success, frame = camera.read()
#     if not success:
#         return jsonify({'error': 'Could not read from camera'})

#     processed_frame, sections = detect_and_classify(frame)

#     conn = get_db_connection()
#     cursor = conn.cursor()

#     fault_count = sum(1 for section in sections if section['fault_name'] != 'Clean')
#     not_fault_count = sum(1 for section in sections if section['fault_name'] == 'Clean')
#     total_loss = sum(section['fault_value'] for section in sections if section['fault_name'] != 'Clean')
#     total_gain = not_fault_count * 20  

#     for section in sections:
#         cursor.execute("""
#             INSERT INTO fault_detect_db1 (fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#         """, (section['fault_name'], section['fault_value'], section['fault_section'], datetime.now(), fault_count, not_fault_count, total_loss, total_gain))

#     conn.commit()
#     cursor.close()
#     conn.close()

#     return jsonify({'status': 'success', 'sections': sections})
    



# @app.route('/exit', methods=['POST'])
# def exit():
#     camera.release()
#     cv2.destroyAllWindows()
#     return "Camera released and window closed", 200

# @app.route('/get_fault_data', methods=['GET'])
# def get_fault_data():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain FROM fault_detect_db1 ORDER BY id DESC LIMIT 20")
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     return jsonify(rows)

# if __name__ == '__main__':
#     app.run(debug=True)


# part2

# import os
# import time
# from datetime import datetime
# from flask import Flask, render_template, Response, request, jsonify, url_for, redirect
# import cv2
# import mysql.connector
# from detection import detect_and_classify

# app = Flask(__name__)
# app.secret_key = 'replace-with-a-random-secret-string'

# # Camera (keep 0 or adjust if you use another camera)
# camera = cv2.VideoCapture(0)

# # MySQL Database configuration — update if necessary
# db_config = {
#     'user': 'root',
#     'password': '134500',
#     'host': 'localhost',
#     'database': 'ranking_db'
# }

# def get_db_connection():
#     conn = mysql.connector.connect(**db_config)
#     return conn

# # In-memory cache for the most recent detection result (simple approach)
# last_detection_data = {}

# @app.route('/')
# def index():
#     """Home page: only live camera feed and Detect button"""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Return streaming response for the live camera feed.
#        The detection used for live feed does not save section images.
#     """
#     def gen_frames():
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             else:
#                 # For live preview we draw bounding boxes and labels but do NOT save section images
#                 processed_frame, _ = detect_and_classify(frame, save_images=False)
#                 ret, buffer = cv2.imencode('.jpg', processed_frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/detect_and_redirect', methods=['POST'])
# def detect_and_redirect():
#     """Capture the current frame, run detection, save section images,
#        insert DB rows and then return JSON with redirect URL to show results.
#     """
#     global last_detection_data

#     success, frame = camera.read()
#     if not success:
#         return jsonify({'error': 'Could not read from camera'}), 500

#     # Perform detection and save section images (save_images=True)
#     processed_frame, sections = detect_and_classify(frame, save_images=True)

#     # Prepare aggregated values
#     fault_count = sum(1 for section in sections if section['fault_name'] != 'Clean')
#     not_fault_count = sum(1 for section in sections if section['fault_name'] == 'Clean')
#     total_loss = sum(section['fault_value'] for section in sections if section['fault_name'] != 'Clean')
#     total_gain = not_fault_count * 20  # you used 20 as gain per clean section

#     # Insert rows into DB (one row per section, mirroring your original behavior)
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         for section in sections:
#             cursor.execute("""
#                 INSERT INTO fault_detect_db1
#                 (fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#             """, (
#                 section['fault_name'],
#                 section['fault_value'],
#                 section['fault_section'],
#                 datetime.now(),
#                 fault_count,
#                 not_fault_count,
#                 total_loss,
#                 total_gain
#             ))
#         conn.commit()
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         print("DB Insert Error:", e)

#     # Compute Health Signal & Suggestion
#     # Logic:
#     #  - 0 faulty sections -> Healthy (Green)
#     #  - 1-2 faulty -> Warning (Yellow)
#     #  - >2 faulty -> Critical (Red)
#     if fault_count == 0:
#         health_status = "Healthy"
#         health_color = "green"
#         suggestion = "Panel looks healthy. No immediate action required."
#     elif fault_count <= 2:
#         health_status = "Warning"
#         health_color = "orange"
#         suggestion = "Minor faults detected. Schedule a cleaning/inspection soon."
#     else:
#         health_status = "Critical"
#         health_color = "red"
#         suggestion = "Multiple faults detected. Immediate maintenance required."

#     # Save last detection to in-memory cache so /fault_predictions can present it
#     last_detection_data = {
#         'timestamp': int(time.time()),
#         'sections': sections,
#         'fault_count': fault_count,
#         'not_fault_count': not_fault_count,
#         'total_loss': total_loss,
#         'total_gain': total_gain,
#         'health_status': health_status,
#         'health_color': health_color,
#         'suggestion': suggestion
#     }

#     # Return redirect URL to client
#     return jsonify({'redirect': url_for('fault_predictions')})

# @app.route('/fault_predictions')
# def fault_predictions():
#     """Render the page that displays the 9 section images and labels for the last detection."""
#     global last_detection_data
#     if not last_detection_data or 'sections' not in last_detection_data:
#         # If no detection yet, go back to home
#         return redirect(url_for('index'))

#     return render_template('fault_predictions.html', data=last_detection_data)

# @app.route('/calculations')
# def calculations():
#     """Render calculation table page with recent DB rows and aggregated data for charts."""
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Last 200 records
#     cursor.execute("""
#         SELECT fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain
#         FROM fault_detect_db1
#         ORDER BY id DESC
#         LIMIT 200
#     """)
#     rows = cursor.fetchall()

#     # Aggregation by fault_name for chart (sum of total_loss and count)
#     cursor.execute("""
#         SELECT fault_name, COUNT(*) AS cnt, SUM(total_loss) AS sum_loss
#         FROM fault_detect_db1
#         GROUP BY fault_name
#     """)
#     agg = cursor.fetchall()

#     cursor.close()
#     conn.close()

#     # Format aggregates for template/chart
#     agg_labels = [row[0] for row in agg]
#     agg_counts = [int(row[1]) for row in agg]
#     agg_losses = [int(row[2] or 0) for row in agg]

#     return render_template('calculations.html',
#                            rows=rows,
#                            agg_labels=agg_labels,
#                            agg_counts=agg_counts,
#                            agg_losses=agg_losses)

# @app.route('/api/fault_rows')
# def api_fault_rows():
#     """Return last DB rows as JSON (for AJAX if needed)."""
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain
#         FROM fault_detect_db1
#         ORDER BY id DESC
#         LIMIT 200
#     """)
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     return jsonify(rows)

# @app.route('/api/agg')
# def api_agg():
#     """Return aggregated data (labels, counts, losses) as JSON for charts."""
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT fault_name, COUNT(*) AS cnt, SUM(total_loss) AS sum_loss
#         FROM fault_detect_db1
#         GROUP BY fault_name
#     """)
#     agg = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     labels = [row[0] for row in agg]
#     counts = [int(row[1]) for row in agg]
#     losses = [int(row[2] or 0) for row in agg]
#     return jsonify({'labels': labels, 'counts': counts, 'losses': losses})

# @app.route('/exit', methods=['POST'])
# def exit_app():
#     """Release camera and cleanup."""
#     try:
#         camera.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print("Error releasing camera:", e)
#     return jsonify({'status': 'camera_released'})

# if __name__ == '__main__':
#     # ensure static/captures exists
#     os.makedirs(os.path.join('static', 'captures'), exist_ok=True)
#     app.run(debug=True)


# part3
import os
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, url_for, redirect
import cv2
import mysql.connector
from detection import detect_and_classify

app = Flask(__name__)
app.secret_key = 'replace-with-a-random-secret-string' #mostly used during llm models

# Camera
camera = cv2.VideoCapture(0) #laptop firs camera 

# MySQL Database config
# changes the root and password used for db connection
db_config = {
    'user': 'root',
    'password': 'Solarix@2025',
    'host': 'localhost',
    'database': 'ranking_db'
}
# security purpopse : we can develop in environmnet of mysql

#helper function not to repeat the connection codeb
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Cache last detection for UI 
last_detection_data = {} #it holds the last detected data cache 

# Fault severity for alerting it is global dictionary
FAULT_SEVERITY = {
    "Electrical-damage": "critical",
    "Physical-Damage": "critical",
    "Dusty": "moderate",
    "Bird-drop": "moderate",
    "Snow-Covered": "moderate",
    "Clean": "none",
    "Unknown": "moderate"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            # Live preview: draw boxes only when panel confidently detected
            processed_frame, _ = detect_and_classify(frame, save_images=False, panel_conf_threshold=0.80)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# it give the health sugestions of the panel
def compute_health_and_suggestion(sections):
    # Any critical fault -> Critical
    has_critical = any(FAULT_SEVERITY.get(s['fault_name'], 'none') == 'critical' for s in sections)
    has_moderate = any(FAULT_SEVERITY.get(s['fault_name'], 'none') == 'moderate' for s in sections)
    # moderate orange color
    # you changes this accordingly
    if has_critical:
        return ("Critical", "red", "Critical faults detected (damage). Immediate maintenance required.")
    if has_moderate:
        return ("Warning", "orange", "Minor faults detected. Schedule cleaning/inspection.")
    return ("Healthy", "green", "Panel looks healthy. No immediate action required.")


# actual code logic
# when you click the detect button then this code logic will came
#return json file
# status code 500 error if not detcted
@app.route('/detect_and_redirect', methods=['POST']) # this is our landing page
def detect_and_redirect():
    global last_detection_data

    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Could not read from camera'}), 500

    # Save images for prediction page
    processed_frame, sections = detect_and_classify(frame, save_images=True, panel_conf_threshold=0.80)
   # sections are labelled as per op

   # the below are metric calculation
   # fault values for non clean sections
    fault_count = sum(1 for s in sections if s['fault_name'] != 'Clean')
    not_fault_count = sum(1 for s in sections if s['fault_name'] == 'Clean') # helthy sections
    total_loss = sum(s['fault_value'] for s in sections if s['fault_name'] != 'Clean')
    # linear gain not faulty * 20 basic it is domain constant it can be adjusted
    total_gain = not_fault_count * 20

    # DB insert (one row per section)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for s in sections:
            cursor.execute("""
                INSERT INTO fault_detect_db1
                (fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                s['fault_name'],
                s['fault_value'],
                s['fault_section'],
                datetime.now(),
                fault_count,
                not_fault_count,
                total_loss,
                total_gain
            ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("DB Insert Error:", e)

    # Health / suggestion based on categories
    health_status, health_color, suggestion = compute_health_and_suggestion(sections)

    # Aggregate categories for alert payload
    by_cat = {} #it is like the cache it will return the json file
    for s in sections:
        by_cat[s['fault_name']] = by_cat.get(s['fault_name'], 0) + 1
# counts how many section fall in category  eg dusty has came 2 times fault came 3 times like this
    last_detection_data = {
        'timestamp': int(time.time()),
        'sections': sections,
        'fault_count': fault_count,
        'not_fault_count': not_fault_count,
        'total_loss': total_loss,
        'total_gain': total_gain,
        'health_status': health_status,
        'health_color': health_color,
        'suggestion': suggestion,
        'by_category': by_cat
    }

    return jsonify({'redirect': url_for('fault_predictions')})

@app.route('/fault_predictions')
#if nothing has been detected it will return to the home otherwise it will render the home.html
def fault_predictions():
    if not last_detection_data or 'sections' not in last_detection_data:
        return redirect(url_for('index'))
    return render_template('fault_predictions.html', data=last_detection_data)

@app.route('/calculations')
# 
def calculations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain
        FROM fault_detect_db1
        ORDER BY id DESC
        LIMIT 200 
    """) # it will have 200 make it 50
    rows = cursor.fetchall()

    cursor.execute("""
        SELECT fault_name, COUNT(*) AS cnt, SUM(total_loss) AS sum_loss
        FROM fault_detect_db1
        GROUP BY fault_name
    """)
    agg = cursor.fetchall()
    cursor.close()
    conn.close()

    agg_labels = [row[0] for row in agg]
    agg_counts = [int(row[1]) for row in agg]
    agg_losses = [int(row[2] or 0) for row in agg] # it will check row againt the null 

    return render_template('calculations.html',
                           rows=rows,
                           agg_labels=agg_labels,
                           agg_counts=agg_counts,
                           agg_losses=agg_losses)
#json api for dashborads 
@app.route('/api/fault_rows')
def api_fault_rows():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fault_name, fault_value, fault_section, t_date, fault_count, no_fault_count, total_loss, total_gain
        FROM fault_detect_db1
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)
#reshaped the arrays for charting for the frontend
@app.route('/api/agg')
def api_agg():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fault_name, COUNT(*) AS cnt, SUM(total_loss) AS sum_loss
        FROM fault_detect_db1
        GROUP BY fault_name
    """)
    agg = cursor.fetchall()
    cursor.close()
    conn.close()
    labels = [row[0] for row in agg]
    counts = [int(row[1]) for row in agg]
    losses = [int(row[2] or 0) for row in agg]
    return jsonify({'labels': labels, 'counts': counts, 'losses': losses})
# last status avaliable
@app.route('/api/last_status')
def api_last_status():
    """Latest health status + suggestion + per-category counts for dashboard alert."""
    if not last_detection_data:
        return jsonify({'available': False})
    payload = last_detection_data.copy()
    payload['available'] = True
    return jsonify(payload)
#it will reduce the camera
@app.route('/exit', methods=['POST'])
def exit_app():
    try:
        camera.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error releasing camera:", e)
    return jsonify({'status': 'camera_released'})

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'captures'), exist_ok=True)
    app.run(debug=True)


# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model # type: ignore



# # # Load models
# # # panel_model = load_model('D:\Data Science All\Live Project\panel_training_code\Classifiers\solar_fault_detection_model.h5')
# # # fault_model = load_model('D:\Data Science All\Live Project\panel_training_code\Classifiers\solar_panel_detection_model.h5')

# panel_model = load_model('solar_fault_detection_model.h5')
# fault_model = load_model('solar_panel_detection_model.h5')


# # Load labels
# with open(r'D:\Data Science All\Live Project\panel_training_code\Classifiers\labels.txt') as f:
#     panel_labels = [line.strip() for line in f]

# with open(r'D:\Data Science All\Live Project\panel_training_code\Classifiers\fault_labels.txt') as f:
#     fault_labels = [line.strip() for line in f]

# # Fault value mapping
# fault_values = {
#     "Bird-drop": 20,
#     "Clean": 0,
#     "Dusty": 20,
#     "Electrical-damage": 20,
#     "Physical-Damage": 20,
#     "Snow-Covered": 20
# }

# def detect_and_classify(frame):
#     height, width, _ = frame.shape
    
#     # Preprocess the frame for panel detection
#     resized_frame = cv2.resize(frame, (150, 150))
#     panel_input = np.expand_dims(resized_frame / 255.0, axis=0)
    
#     # Predict panel presence
#     panel_pred = panel_model.predict(panel_input)
#     panel_class = panel_labels[np.argmax(panel_pred)]
    
#     sections = []

#     if panel_class == "Solar_Panel":
#         cv2.putText(frame, "Solar Panel Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # Create a green bounding box around the panel
#         cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 2)
        
#         # Divide the panel into 9 sections and detect faults
#         section_height = height // 3
#         section_width = width // 3
        
#         for i in range(3):
#             for j in range(3):
#                 x1, y1 = j * section_width, i * section_height
#                 x2, y2 = x1 + section_width, y1 + section_height
#                 section = frame[y1:y2, x1:x2]
                
#                 # Preprocess section for fault detection
#                 resized_section = cv2.resize(section, (150, 150))
#                 section_input = np.expand_dims(resized_section / 255.0, axis=0)
                
#                 # Predict fault in the section
#                 fault_pred = fault_model.predict(section_input)
#                 fault_class = fault_labels[np.argmax(fault_pred)]
                
#                 # Draw red bounding box and label for each section
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, fault_class, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
#                 # Store section details
#                 section_id = f"Section_{i}_{j}"
#                 sections.append({
#                     'fault_name': fault_class,
#                     'fault_value': fault_values[fault_class],
#                     'fault_section': section_id
#                 })
    
#     return frame, sections


# def detect_and_classify(frame):
#     height, width, _ = frame.shape

#     # Preprocess the frame for panel detection
#     resized_frame = cv2.resize(frame, (150, 150))
#     panel_input = np.expand_dims(resized_frame / 255.0, axis=0)

#     try:
#         # Predict panel presence
#         panel_pred = panel_model.predict(panel_input)
#         panel_class = panel_labels[np.argmax(panel_pred)]
#     except IndexError:
#         panel_class = "Unknown"
#         print("Error: Panel prediction index out of range.")
#         return frame, []

#     sections = []

#     if panel_class == "Solar_Panel":
#         cv2.putText(frame, "Solar Panel Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Create a green bounding box around the panel
#         cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)

#         # Divide the panel into 9 sections and detect faults
#         section_height = height // 3
#         section_width = width // 3

#         for i in range(3):
#             for j in range(3):
#                 x1, y1 = j * section_width, i * section_height
#                 x2, y2 = x1 + section_width, y1 + section_height
#                 section = frame[y1:y2, x1:x2]

#                 # Preprocess section for fault detection
#                 resized_section = cv2.resize(section, (150, 150))
#                 section_input = np.expand_dims(resized_section / 255.0, axis=0)

#                 try:
#                     # Predict fault in the section
#                     fault_pred = fault_model.predict(section_input)
#                     fault_class = fault_labels[np.argmax(fault_pred)]
#                 except IndexError:
#                     fault_class = "Unknown"
#                     print("Error: Fault prediction index out of range.")

#                 # Draw red bounding box and label for each section
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, fault_class, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 # Store section details
#                 section_id = f"Section_{i}_{j}"
#                 sections.append({
#                     'fault_name': fault_class,
#                     'fault_value': fault_values.get(fault_class, 0),
#                     'fault_section': section_id
#                 })

#     return frame, sections



# def detect_and_classify(frame):
#     height, width, _ = frame.shape
#     resized_frame = cv2.resize(frame, (150, 150))
#     panel_input = np.expand_dims(resized_frame / 255.0, axis=0)

#     try:
#         panel_pred = panel_model.predict(panel_input)
#         panel_class_idx = np.argmax(panel_pred)
        
#         if panel_class_idx >= len(panel_labels):  # Prevent index error
#             print("Error: Panel prediction index out of range.")
#             return frame, []

#         panel_class = panel_labels[panel_class_idx]

#     except Exception as e:
#         print(f"Error in panel prediction: {e}")
#         return frame, []

#     sections = []

#     if panel_class == "Solar_Panel":
#         cv2.putText(frame, "Solar Panel Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)

#         section_height = height // 3
#         section_width = width // 3

#         for i in range(3):
#             for j in range(3):
#                 x1, y1 = j * section_width, i * section_height
#                 x2, y2 = x1 + section_width, y1 + section_height
#                 section = frame[y1:y2, x1:x2]

#                 resized_section = cv2.resize(section, (150, 150))
#                 section_input = np.expand_dims(resized_section / 255.0, axis=0)

#                 try:
#                     fault_pred = fault_model.predict(section_input)
#                     fault_class_idx = np.argmax(fault_pred)
                    
#                     if fault_class_idx >= len(fault_labels):  # Prevent index error
#                         print("Error: Fault prediction index out of range.")
#                         fault_class = "Unknown"
#                     else:
#                         fault_class = fault_labels[fault_class_idx]

#                 except Exception as e:
#                     print(f"Error in fault prediction: {e}")
#                     fault_class = "Unknown"

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, fault_class, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 section_id = f"Section_{i}_{j}"
#                 sections.append({
#                     'fault_name': fault_class,
#                     'fault_value': fault_values.get(fault_class, 0),
#                     'fault_section': section_id
#                 })

#     return frame, sections



# part2

# import os
# import time
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Paths: models placed in ./models/
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, 'models')

# PANEL_MODEL_PATH = os.path.join(MODELS_DIR, 'solar_panel_detection_model.h5')
# FAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'solar_fault_detection_model.h5')
# LABELS_PATH = os.path.join(MODELS_DIR, 'labels.txt')
# FAULT_LABELS_PATH = os.path.join(MODELS_DIR, 'fault_labels.txt')

# # Load models (will raise if files missing)
# try:
#     panel_model = load_model(PANEL_MODEL_PATH)
# except Exception as e:
#     print("Error loading panel model:", e)
#     panel_model = None

# try:
#     fault_model = load_model(FAULT_MODEL_PATH)
# except Exception as e:
#     print("Error loading fault model:", e)
#     fault_model = None

# # Load labels
# if os.path.exists(LABELS_PATH):
#     with open(LABELS_PATH, 'r') as f:
#         panel_labels = [line.strip() for line in f]
# else:
#     panel_labels = ['Solar_Panel', 'No_Panel']  # fallback

# if os.path.exists(FAULT_LABELS_PATH):
#     with open(FAULT_LABELS_PATH, 'r') as f:
#         fault_labels = [line.strip() for line in f]
# else:
#     fault_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# # Fault value mapping
# fault_values = {
#     "Bird-drop": 20,
#     "Clean": 0,
#     "Dusty": 20,
#     "Electrical-damage": 20,
#     "Physical-Damage": 20,
#     "Snow-Covered": 20,
#     # default: 0
# }

# def detect_and_classify(frame, save_images=False, save_base_dir=os.path.join('static', 'captures')):
#     """
#     Detect if frame contains a solar panel and classify 3x3 sections.
#     If save_images=True, saves each section into static/captures/d_<timestamp>/section_i_j.jpg
#     Returns (annotated_frame, sections_list)
#     Each section dict: {'fault_name', 'fault_value', 'fault_section', 'image' (relative path inside static/)}
#     """
#     height, width, _ = frame.shape
#     # Preserve original frame copy for drawing
#     out_frame = frame.copy()

#     # Prepare input for panel classifier
#     resized_frame = cv2.resize(frame, (150, 150))
#     panel_input = np.expand_dims(resized_frame / 255.0, axis=0)

#     try:
#         if panel_model is None:
#             raise RuntimeError("Panel model not loaded")
#         panel_pred = panel_model.predict(panel_input)
#         panel_class_idx = int(np.argmax(panel_pred))
#         if panel_class_idx >= len(panel_labels):
#             print("Panel prediction index out of range")
#             return out_frame, []
#         panel_class = panel_labels[panel_class_idx]
#     except Exception as e:
#         print("Error in panel prediction:", e)
#         return out_frame, []

#     sections = []

#     if panel_class == "Solar_Panel":
#         cv2.putText(out_frame, "Solar Panel Detected", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.rectangle(out_frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)

#         section_height = height // 3
#         section_width = width // 3

#         timestamp = int(time.time())
#         folder_name = f"d_{timestamp}"
#         save_folder = os.path.join(save_base_dir, folder_name)
#         if save_images:
#             os.makedirs(save_folder, exist_ok=True)

#         idx = 0
#         for i in range(3):
#             for j in range(3):
#                 x1, y1 = j * section_width, i * section_height
#                 x2, y2 = x1 + section_width, y1 + section_height
#                 section_img = frame[y1:y2, x1:x2]

#                 # Resize and predict fault for this section
#                 try:
#                     resized_section = cv2.resize(section_img, (150, 150))
#                     section_input = np.expand_dims(resized_section / 255.0, axis=0)
#                     if fault_model is None:
#                         raise RuntimeError("Fault model not loaded")
#                     fault_pred = fault_model.predict(section_input)
#                     fault_class_idx = int(np.argmax(fault_pred))
#                     if fault_class_idx >= len(fault_labels):
#                         fault_class = "Unknown"
#                     else:
#                         fault_class = fault_labels[fault_class_idx]
#                 except Exception as e:
#                     print("Error in fault prediction:", e)
#                     fault_class = "Unknown"

#                 # Draw rectangle and label
#                 cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(out_frame, fault_class, (x1 + 10, y1 + 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 # Save section image if requested
#                 image_rel_path = None
#                 if save_images:
#                     filename = f"section_{i}_{j}.jpg"
#                     save_path = os.path.join(save_folder, filename)
#                     # Save the raw cropped section (you can save annotated section if needed)
#                     cv2.imwrite(save_path, section_img)
#                     # For template usage, store path relative to static/
#                     image_rel_path = os.path.join('captures', folder_name, filename).replace(os.sep, '/')

#                 sections.append({
#                     'fault_name': fault_class,
#                     'fault_value': int(fault_values.get(fault_class, 0)),
#                     'fault_section': f"Section_{i}_{j}",
#                     'image': image_rel_path  # may be None if save_images=False
#                 })
#                 idx += 1

#     else:
#         # No panel detected -> just inform
#         cv2.putText(out_frame, "No Solar Panel Detected", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return out_frame, sections

# part3

import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model #load the pretrained deep learning models from h5 files

# Paths: models placed in ./models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models') #it is abs path of cuurent file 
#paths of current files
PANEL_MODEL_PATH = os.path.join(MODELS_DIR, 'solar_panel_detection_model.h5')
FAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'solar_fault_detection_model.h5')
LABELS_PATH = os.path.join(MODELS_DIR, 'labels.txt')
FAULT_LABELS_PATH = os.path.join(MODELS_DIR, 'fault_labels.txt')

# Load models loading the solar panel cnn model
try:
    panel_model = load_model(PANEL_MODEL_PATH)
except Exception as e:
    print("Error loading panel model:", e)
    panel_model = None
# same for falut models
try:
    fault_model = load_model(FAULT_MODEL_PATH)
except Exception as e:
    print("Error loading fault model:", e)
    fault_model = None

# Load labels read the text files 
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        panel_labels = [line.strip() for line in f]
else:
    panel_labels = ['Solar_Panel', 'No_Panel']  # fallback

if os.path.exists(FAULT_LABELS_PATH):
    with open(FAULT_LABELS_PATH, 'r') as f:
        fault_labels = [line.strip() for line in f]
else:
    fault_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Fault value mapping assign numeric impact value for faults
fault_values = {
    "Bird-drop": 20,
    "Clean": 0,
    "Dusty": 20,
    "Electrical-damage": 20,
    "Physical-Damage": 20,
    "Snow-Covered": 20,
}
# frame imaging camera and video feeding save save_image to store crop images 
def detect_and_classify(frame, save_images=False, save_base_dir=os.path.join('static', 'captures'),
                        panel_conf_threshold=0.80):
    """
    Detect if frame contains a solar panel and classify 3x3 sections.
    If save_images=True, saves each section into static/captures/d_<timestamp>/section_i_j.jpg
    Returns (annotated_frame, sections_list)
    Each section dict: {'fault_name', 'fault_value', 'fault_section', 'image'(optional relative path)}
    """
    height, width, _ = frame.shape
    out_frame = frame.copy()

    # Prepare input for panel classifier
    resized_frame = cv2.resize(frame, (150, 150)) #size of frame change the size 
    panel_input = np.expand_dims(resized_frame / 255.0, axis=0) #it is standrad 
# ensure only solara panel with conf grater than 80
    try:
        if panel_model is None:
            raise RuntimeError("Panel model not loaded")
        panel_pred = panel_model.predict(panel_input, verbose=0)
        panel_class_idx = int(np.argmax(panel_pred))
        panel_conf = float(np.max(panel_pred))
        if panel_class_idx >= len(panel_labels):
            print("Panel prediction index out of range")
            return out_frame, []
        panel_class = panel_labels[panel_class_idx]
    except Exception as e:
        print("Error in panel prediction:", e)
        return out_frame, []

    sections = []

    # Only draw grid if the class is Solar_Panel AND confidence >= threshold
    if panel_class == "Solar_Panel" and panel_conf >= panel_conf_threshold:
        cv2.putText(out_frame, f"Solar Panel Detected ({panel_conf:.2f})", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(out_frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)

        section_height = height // 3
        section_width = width // 3

        timestamp = int(time.time())
        folder_name = f"d_{timestamp}"
        save_folder = os.path.join(save_base_dir, folder_name)
        if save_images:
            os.makedirs(save_folder, exist_ok=True)

        for i in range(3):
            for j in range(3):
                x1, y1 = j * section_width, i * section_height
                x2, y2 = x1 + section_width, y1 + section_height
                section_img = frame[y1:y2, x1:x2]

                # Predict fault for this section
                try:
                    resized_section = cv2.resize(section_img, (150, 150))
                    section_input = np.expand_dims(resized_section / 255.0, axis=0)
                    if fault_model is None:
                        raise RuntimeError("Fault model not loaded")
                    fault_pred = fault_model.predict(section_input, verbose=0)
                    fault_class_idx = int(np.argmax(fault_pred))
                    if fault_class_idx >= len(fault_labels):
                        fault_class = "Unknown"
                    else:
                        fault_class = fault_labels[fault_class_idx]
                except Exception as e:
                    print("Error in fault prediction:", e)
                    fault_class = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(out_frame, fault_class, (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                image_rel_path = None
                if save_images:
                    filename = f"section_{i}_{j}.jpg"
                    save_path = os.path.join(save_folder, filename)
                    cv2.imwrite(save_path, section_img)
                    image_rel_path = os.path.join('captures', folder_name, filename).replace(os.sep, '/')

                sections.append({
                    'fault_name': fault_class,
                    'fault_value': int(fault_values.get(fault_class, 0)),
                    'fault_section': f"Section_{i}_{j}",
                    'image': image_rel_path
                })
    else:
        # No confident panel detection a msg if detection confidence is too low
        msg = "No Solar Panel Detected" if panel_class != "Solar_Panel" else f"No Panel (low conf {panel_conf:.2f})"
        cv2.putText(out_frame, msg, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return out_frame, sections


import cv2
import easyocr
import pyttsx3
import threading
import time
import numpy as np
import torch

class CameraOCRApp:
    def __init__(self):
        # Auto-detect CUDA GPU for PyTorch/EasyOCR
        self.use_gpu = torch.cuda.is_available()
        print(f"[INFO] CUDA GPU acceleration available: {self.use_gpu}")
        print("[INFO] Loading EasyOCR Model... Please wait (this may take a few seconds on first run)...")
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        print("[INFO] EasyOCR model loaded successfully.")

        # Thread-safe state variables
        self.lock = threading.Lock()
        self.status_message = "Ready. Position text and press SPACE to scan."
        self.detected_text = ""
        self.boxes = []
        self.result_timestamp = 0.0
        self.is_processing = False
        self.is_running = True

    def speak(self, text):
        """Runs the text-to-speech engine in a background thread to prevent GUI freezing."""
        def tts_target():
            try:
                # Initialize pyttsx3 inside the thread to avoid COM apartment issues on Windows
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speaking rate (words per minute)
                
                # Retrieve available voices and select a default if needed
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)
                
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[ERROR] TTS Thread error: {e}")
                
        threading.Thread(target=tts_target, daemon=True).start()

    def process_ocr(self, frame_to_process):
        """Processes OCR on a captured frame in a background thread."""
        with self.lock:
            self.is_processing = True
            self.status_message = "Processing image and reading text..."

        start_time = time.time()
        try:
            # Convert to grayscale for OCR efficiency
            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            
            # Run EasyOCR
            results = self.reader.readtext(gray)
            
            # Parse results
            text_list = []
            boxes_list = []
            
            for (bbox, text, prob) in results:
                if prob > 0.35:  # Confidence threshold to filter out OCR noise
                    text_list.append(text)
                    # Convert bounding box coordinates to integer tuples
                    box_points = np.array(bbox, dtype=np.int32)
                    boxes_list.append((box_points, text))
            
            detected = " ".join(text_list).strip()
            elapsed_time = time.time() - start_time
            print(f"[OCR Done] Detected: '{detected}' (took {elapsed_time:.2f}s)")
            
            with self.lock:
                self.detected_text = detected
                self.boxes = boxes_list
                self.result_timestamp = time.time()
                self.is_processing = False
                
                if detected:
                    self.status_message = f"Detected: {detected}"
                    self.speak(detected)
                else:
                    self.status_message = "No clear text detected. Try again."
                    
        except Exception as e:
            with self.lock:
                self.is_processing = False
                self.status_message = f"OCR Error: {str(e)}"
                print(f"[ERROR] OCR Thread error: {e}")

    def run(self):
        # Open webcam
        print("[INFO] Starting webcam...")
        cap = cv2.VideoCapture(0)
        
        # Adjust camera resolution if supported (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[ERROR] Could not open webcam index 0. Checking index 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("[ERROR] No webcam found. Please connect a camera and try again.")
                return

        window_name = "Camera Real-Time OCR Reader"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("[INFO] Camera stream active. Press SPACE to capture/read text. Press Q to exit.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # Create a copy of the frame to draw overlays on
            draw_frame = frame.copy()
            h, w, c = draw_frame.shape

            # Overlay Bounding Boxes if they were detected within the last 4 seconds
            current_time = time.time()
            with self.lock:
                show_boxes = (current_time - self.result_timestamp) < 4.0
                is_proc = self.is_processing
                status = self.status_message
                boxes = list(self.boxes)

            if show_boxes and not is_proc:
                for box_points, text in boxes:
                    # Draw polygon bounding box around text
                    cv2.polylines(draw_frame, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Put detected text label above the bounding box
                    top_left = tuple(box_points[0])
                    # Ensure text is not drawn off-screen
                    text_y = max(top_left[1] - 10, 20)
                    cv2.putText(draw_frame, text, (top_left[0], text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Create translucent overlay bar at the top for instructions
            overlay = draw_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
            
            # Create translucent overlay bar at the bottom for status
            bar_height = 50
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (30, 30, 30), -1)
            
            # Apply translucent overlays
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0, draw_frame)

            # Put overlay texts
            instructions_text = "Press SPACE to Scan Text | Press Q to Exit"
            cv2.putText(draw_frame, instructions_text, (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            # Prepend processing tag if OCR is active
            display_status = f"[SCANNING...] {status}" if is_proc else status
            cv2.putText(draw_frame, display_status, (15, h - 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if not is_proc else (0, 255, 255), 1, cv2.LINE_AA)

            # Display the frame
            cv2.imshow(window_name, draw_frame)

            # Handle keypress events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC key
                self.is_running = False
            elif key == ord(' '):  # SPACE key
                if not is_proc:
                    print("[INFO] Capturing frame for OCR...")
                    # Send copy of original frame to avoid drawing overlays in OCR engine
                    ocr_thread = threading.Thread(target=self.process_ocr, args=(frame.copy(),), daemon=True)
                    ocr_thread.start()

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")

if __name__ == "__main__":
    app = CameraOCRApp()
    app.run()

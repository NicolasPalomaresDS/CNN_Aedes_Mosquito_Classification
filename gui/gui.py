import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

class CNNClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aedes Classifier")
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # Load pre-trained model
        model_path = "../model/model.keras" 
        self.model = load_model(model_path)
        
        # Class labels
        self.class_labels = ['aegypti', 'albopictus']
        
        # Title
        title_label = tk.Label(
            root, 
            text="Aedes Classifier Demo", 
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=15
        )
        title_label.pack(fill=tk.X)
        
        # Main container
        main_frame = tk.Frame(root, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Image display
        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(
            left_frame, 
            text="No image loaded\n\nClick 'Load Image' to begin",
            font=("Arial", 12),
            bg="#ecf0f1",
            relief=tk.RIDGE,
            borderwidth=2
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Load button
        load_btn = tk.Button(
            left_frame,
            text="Load Image",
            command=self.load_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        load_btn.pack(pady=(15, 5))
        
        # Right side - Prediction result
        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.result_label = tk.Label(
            right_frame,
            text="Waiting for image...",
            font=("Arial", 16),
            bg="#ecf0f1",
            relief=tk.RIDGE,
            borderwidth=2,
            wraplength=350,
            justify=tk.CENTER,
            padx=20,
            pady=30
        )
        self.result_label.pack(fill=tk.BOTH, expand=True)
        
        self.current_image = None
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                img = Image.open(file_path)
                self.current_image = img.copy()
                
                # Resize for display (maintain aspect ratio)
                display_img = img.copy()
                display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_img)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Make prediction
                self.predict_image(img)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def predict_image(self, img):
        try:
            # Prepare image for model
            img_resized = img.resize((224, 224))
            
            # Convert to RGB if image is in different mode
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Use EfficientNetV2 preprocessing (scales to [-1, 1])
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get all predictions for debugging
            print("Raw predictions:", predictions[0])  # Debug output
            
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # Show top predictions
            result_text = "Predictions:\n\n"
            for idx in range(len(self.class_labels)):
                result_text += f"{self.class_labels[idx]}\n"
                result_text += f"{predictions[0][idx] * 100:.2f}%\n\n"
            
            self.result_label.config(
                text=result_text,
                font=("Arial", 18, "bold"),
                fg="#2c3e50"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

def main():
    root = tk.Tk()
    app = CNNClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
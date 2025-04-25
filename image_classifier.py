import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("تطبيق تصنيف الصور")
        self.root.geometry("600x450")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        # تحميل نموذج CLIP
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.categories = {
            'أشخاص': "a photo of a person",
            'طعام': "a photo of food",
            'حيوانات': "a photo of an animal",
            'نباتات': "a photo of a plant",
            'طبيعة': "a photo of a landscape",
            'مباني': "a photo of a building",
            'مركبات': "a photo of a vehicle",
            'أخرى': "a miscellaneous photo"
        }

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(main_frame, text="تطبيق تصنيف الصور باستخدام CLIP", 
                               font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        input_frame = tk.LabelFrame(main_frame, text="اختيار المجلدات", font=("Arial", 12), bg="#f0f0f0")
        input_frame.pack(fill=tk.X, pady=10)

        # مجلد المصدر
        source_frame = tk.Frame(input_frame, bg="#f0f0f0")
        source_frame.pack(fill=tk.X, pady=5)
        source_label = tk.Label(source_frame, text="مجلد الصور المصدر:", width=15, anchor="w", bg="#f0f0f0")
        source_label.pack(side=tk.LEFT, padx=5)
        self.source_entry = tk.Entry(source_frame, width=40)
        self.source_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        source_button = tk.Button(source_frame, text="استعراض", command=self.browse_source)
        source_button.pack(side=tk.LEFT, padx=5)

        # مجلد الوجهة
        dest_frame = tk.Frame(input_frame, bg="#f0f0f0")
        dest_frame.pack(fill=tk.X, pady=5)
        dest_label = tk.Label(dest_frame, text="مجلد الوجهة:", width=15, anchor="w", bg="#f0f0f0")
        dest_label.pack(side=tk.LEFT, padx=5)
        self.dest_entry = tk.Entry(dest_frame, width=40)
        self.dest_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        dest_button = tk.Button(dest_frame, text="استعراض", command=self.browse_dest)
        dest_button.pack(side=tk.LEFT, padx=5)

        # زر بدء التصنيف
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        self.start_button = tk.Button(button_frame, text="بدء تصنيف الصور", 
                                      command=self.start_classification, width=20, height=2,
                                      bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.start_button.pack()

        # شريط التقدم
        progress_frame = tk.Frame(main_frame, bg="#f0f0f0")
        progress_frame.pack(fill=tk.X, pady=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100)
        self.progress_bar.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(progress_frame, text="جاهز", bg="#f0f0f0")
        self.status_label.pack(pady=5)

        # سجل العمليات
        log_frame = tk.LabelFrame(main_frame, text="سجل العمليات", font=("Arial", 12), bg="#f0f0f0")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.log_text = tk.Text(log_frame, height=8, width=50, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def browse_source(self):
        folder_path = filedialog.askdirectory(title="اختر مجلد الصور المصدر")
        if folder_path:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, folder_path)

    def browse_dest(self):
        folder_path = filedialog.askdirectory(title="اختر مجلد الوجهة")
        if folder_path:
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, folder_path)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def update_status(self, message):
        self.status_label.config(text=message)

    def classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        text_inputs = list(self.categories.values())

        inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze()

        best_idx = torch.argmax(probs).item()
        category = list(self.categories.keys())[best_idx]
        return category, probs[best_idx].item()

    def start_classification(self):
        source_dir = self.source_entry.get()
        dest_dir = self.dest_entry.get()

        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("خطأ", "الرجاء تحديد مجلد المصدر بشكل صحيح")
            return

        if not dest_dir or not os.path.isdir(dest_dir):
            messagebox.showerror("خطأ", "الرجاء تحديد مجلد الوجهة بشكل صحيح")
            return

        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.classification_thread, args=(source_dir, dest_dir)).start()

    def classification_thread(self, source_dir, dest_dir):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(image_extensions)]

        total_images = len(image_files)
        if total_images == 0:
            messagebox.showinfo("معلومات", "لم يتم العثور على صور في المجلد المحدد")
            self.start_button.config(state=tk.NORMAL)
            return

        self.log_message(f"تم العثور على {total_images} صورة")

        for cat in self.categories:
            os.makedirs(os.path.join(dest_dir, cat), exist_ok=True)

        for i, filename in enumerate(image_files):
            image_path = os.path.join(source_dir, filename)
            category, prob = self.classify_image(image_path)
            shutil.copy2(image_path, os.path.join(dest_dir, category, filename))

            self.progress_var.set((i + 1) / total_images * 100)
            self.update_status(f"تم تصنيف: {filename}")
            self.log_message(f"{filename} => {category} ({prob:.2%})")

        self.progress_var.set(100)
        self.update_status("اكتمل التصنيف!")
        self.log_message("اكتمل تصنيف جميع الصور")
        messagebox.showinfo("اكتمل", "تم تصنيف جميع الصور بنجاح!")
        self.start_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

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
    """
    تطبيق تصنيف الصور باستخدام الذكاء الاصطناعي
    يقوم بتصنيف الصور إلى فئات محددة مسبقًا باستخدام نموذج CLIP من OpenAI
    """
    def __init__(self, root):
        """
        دالة الإنشاء للتطبيق
        
        المعلمات:
            root: نافذة tkinter الرئيسية
        """
        # تهيئة النافذة الرئيسية
        self.root = root
        self.root.title("تصنيف الصور الذكي باستخدام الذكاء الاصطناعي")
        self.root.geometry("600x500")  # الحجم الافتراضي للنافذة
        self.root.configure(bg="#F5F6FA")  # لون خلفية التطبيق
        self.root.resizable(True, True)  # السماح بتغيير حجم النافذة
        
        # تعيين الوزن النسبي للصفوف عند تغيير حجم النافذة
        self.root.rowconfigure(0, weight=3)  # المنطقة العلوية (75% من الارتفاع)
        self.root.rowconfigure(1, weight=1)  # منطقة السجل (25% من الارتفاع)
        self.root.columnconfigure(0, weight=1)  # عمود واحد يمتد بعرض النافذة

        # تعريف الألوان المستخدمة في التطبيق
        self.colors = {
            "bg": "#F5F6FA",           # لون الخلفية الرئيسية
            "primary": "#3B82F6",      # اللون الأساسي (أزرق)
            "secondary": "#FB923C",    # اللون الثانوي (برتقالي فاتح)
            "light_blue": "#60A5FA",   # أزرق فاتح لشريط التقدم
            "light_gray": "#E5E7EB"    # رمادي فاتح للخلفيات الثانوية
        }

        # تحميل نموذج CLIP للتصنيف
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # تعريف فئات التصنيف (بالعربية والإنجليزية)
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

        # إنشاء عناصر واجهة المستخدم
        self.create_widgets()

    def create_widgets(self):
        """
        إنشاء جميع عناصر واجهة المستخدم في التطبيق
        """
        # نمط موحد للعناصر النصية
        style = {
            "bg": self.colors["bg"],  # لون الخلفية
            "font": ("Cairo", 12),  # الخط ونوعه
            "anchor": "e",  # محاذاة النص إلى اليمين
            "justify": "right"  # اتجاه النص من اليمين إلى اليسار
        }

        # الإطار الرئيسي العلوي يحتوي على كل شيء ماعدا سجل العمليات
        top_frame = tk.Frame(self.root, bg=self.colors["bg"])
        top_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 5))
        top_frame.columnconfigure(0, weight=1)  # توسيع العناصر الداخلية أفقياً

        # عنوان التطبيق
        title_label = tk.Label(top_frame, text="تصنيف الصور الذكي باستخدام الذكاء الاصطناعي",
                               font=("Cairo", 16, "bold"), bg=self.colors["bg"], fg=self.colors["primary"], anchor="center")
        title_label.pack(pady=5)

        # إطار اختيار المجلدات
        input_frame = tk.LabelFrame(top_frame, text="اختيار المجلدات", font=("Cairo", 12, "bold"),
                                    bg=self.colors["bg"], fg=self.colors["primary"], bd=2, labelanchor="ne")
        input_frame.pack(fill=tk.X, pady=5)

        # إطار اختيار مجلد الصور المصدر
        source_frame = tk.Frame(input_frame, bg=self.colors["bg"])
        source_frame.pack(fill=tk.X, pady=3)
        source_label = tk.Label(source_frame, text=": مجلد الصور", width=15, **style)
        source_label.pack(side=tk.RIGHT, padx=5)
        self.source_entry = tk.Entry(source_frame, width=40, justify="right", 
                                    highlightbackground=self.colors["light_gray"], 
                                    highlightthickness=1)
        self.source_entry.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        source_button = tk.Button(source_frame, text="استعراض", 
                                 command=self.browse_source,
                                 bg=self.colors["light_blue"], fg="white",
                                 activebackground=self.colors["light_blue"],
                                 activeforeground="white")
        source_button.pack(side=tk.RIGHT, padx=5)

        # إطار اختيار مجلد الوجهة
        dest_frame = tk.Frame(input_frame, bg=self.colors["bg"])
        dest_frame.pack(fill=tk.X, pady=3)
        dest_label = tk.Label(dest_frame, text=": مجلد الوجهة", width=15, **style)
        dest_label.pack(side=tk.RIGHT, padx=5)
        self.dest_entry = tk.Entry(dest_frame, width=40, justify="right",
                                 highlightbackground=self.colors["light_gray"], 
                                 highlightthickness=1)
        self.dest_entry.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        dest_button = tk.Button(dest_frame, text="استعراض", 
                              command=self.browse_dest,
                              bg=self.colors["light_blue"], fg="white",
                              activebackground=self.colors["light_blue"],
                              activeforeground="white")
        dest_button.pack(side=tk.RIGHT, padx=5)

        # إطار زر بدء التصنيف
        button_frame = tk.Frame(top_frame, bg=self.colors["bg"])
        button_frame.pack(pady=10)
        self.start_button = tk.Button(button_frame, text="بدء تصنيف الصور",
                                      command=self.start_classification, width=20, height=1,
                                      bg=self.colors["primary"], fg="white", font=("Cairo", 12, "bold"),
                                      activebackground=self.colors["light_blue"], activeforeground="white", 
                                      disabledforeground="white")
        self.start_button.pack()

        # إطار شريط التقدم وحالة التصنيف
        progress_frame = tk.Frame(top_frame, bg=self.colors["bg"])
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_var = tk.DoubleVar()  # متغير لتخزين قيمة التقدم (0-100)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100, style="blue.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(progress_frame, text="جاهز", bg=self.colors["bg"], fg=self.colors["primary"], 
                                    font=("Cairo", 11), anchor="e", justify="right")
        self.status_label.pack(pady=2, anchor="e")

        # إطار سجل العمليات (منفصل عن الإطار العلوي لضمان ظهوره)
        log_frame = tk.LabelFrame(self.root, text="سجل العمليات", font=("Cairo", 12, "bold"),
                                bg=self.colors["bg"], fg=self.colors["primary"], bd=2, labelanchor="ne")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # ضبط الإطار الداخلي للسجل
        log_inner_frame = tk.Frame(log_frame, bg=self.colors["bg"])
        log_inner_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_inner_frame.columnconfigure(0, weight=1)  # توسيع النص أفقياً
        log_inner_frame.rowconfigure(0, weight=1)     # توسيع النص رأسياً

        # إضافة عنصر النص وشريط التمرير لسجل العمليات
        self.log_text = tk.Text(log_inner_frame, wrap=tk.WORD, font=("Cairo", 11), height=3,
                              highlightbackground=self.colors["light_gray"], highlightthickness=1) 
        self.log_scrollbar = tk.Scrollbar(log_inner_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        
        # وضع عناصر السجل باستخدام نظام grid لضمان التوسيع السليم
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")  # شريط التمرير يمتد رأسياً
        self.log_text.grid(row=0, column=0, sticky="nsew")     # النص يمتد في جميع الاتجاهات

        # تكوين وسوم للنص المميز وللنص العربي (RTL)
        self.log_text.tag_configure("highlight", foreground=self.colors["light_blue"])
        self.log_text.tag_configure("rtl", justify="right", lmargin1=20, lmargin2=20, rmargin=5)
        self.log_text.tag_configure("rtl_highlight", foreground=self.colors["light_blue"], justify="right", lmargin1=20, lmargin2=20, rmargin=5)
        
        # ضبط النص العربي بالكامل ليكون RTL
        self.log_text.configure(wrap="word", font=("Cairo", 11), height=3, highlightbackground=self.colors["light_gray"], 
                              highlightthickness=1, relief="flat")
        
        # إضافة بعض النص الافتراضي لمنطقة السجل (باستخدام وسم RTL)
        self.log_text.insert(tk.END, "جاهز لاستقبال الصور...\n", "rtl_highlight")
        self.log_text.insert(tk.END, "يرجى اختيار مجلد الصور ومجلد الوجهة ثم الضغط على زر بدء تصنيف الصور\n", "rtl")

        # تعيين نمط شريط التقدم
        style = ttk.Style()
        style.theme_use('default')
        style.configure("blue.Horizontal.TProgressbar", 
                        troughcolor=self.colors["light_gray"], 
                        background=self.colors["light_blue"], 
                        thickness=20)

    def browse_source(self):
        """
        فتح مربع حوار لاختيار مجلد الصور المصدر
        """
        folder_path = filedialog.askdirectory(title="اختر مجلد الصور")
        if folder_path:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, folder_path)

    def browse_dest(self):
        """
        فتح مربع حوار لاختيار مجلد الوجهة
        """
        folder_path = filedialog.askdirectory(title="اختر مجلد الوجهة")
        if folder_path:
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, folder_path)

    def log_message(self, message, highlight=False):
        """
        إضافة رسالة إلى سجل العمليات
        
        المعلمات:
            message: النص المراد إضافته إلى السجل
            highlight: إذا كان True، سيتم تمييز النص بلون مختلف
        """
        if highlight:
            self.log_text.insert(tk.END, message + "\n", "rtl_highlight")
        else:
            self.log_text.insert(tk.END, message + "\n", "rtl")
        self.log_text.see(tk.END)  # تمرير السجل لرؤية آخر رسالة

    def update_status(self, message):
        """
        تحديث نص حالة التصنيف
        
        المعلمات:
            message: نص الحالة الجديد
        """
        self.status_label.config(text=message)

    def classify_image(self, image_path):
        """
        تصنيف صورة واحدة باستخدام نموذج CLIP
        
        المعلمات:
            image_path: مسار الصورة المراد تصنيفها
            
        الإرجاع:
            category: الفئة المصنفة (اسم الفئة بالعربية)
            prob: درجة الثقة في التصنيف (قيمة بين 0 و 1)
        """
        # فتح الصورة وتحويلها إلى الصيغة المناسبة للنموذج
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        
        # تجهيز النصوص المقابلة للفئات
        text_inputs = list(self.categories.values())
        
        # تجهيز المدخلات للنموذج
        inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        
        # تشغيل النموذج للحصول على التوقعات
        outputs = self.model(**inputs)
        
        # حساب احتمالات الفئات
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        
        # اختيار الفئة ذات أعلى احتمال
        best_idx = torch.argmax(probs).item()
        category = list(self.categories.keys())[best_idx]
        
        return category, probs[best_idx].item()

    def start_classification(self):
        """
        بدء عملية تصنيف الصور بعد التحقق من صحة المجلدات
        """
        # الحصول على مسارات المجلدات من حقول الإدخال
        source_dir = self.source_entry.get()
        dest_dir = self.dest_entry.get()

        # التحقق من صحة مجلد المصدر
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("خطأ", "يرجى اختيار مجلد الصور الصحيح")
            return

        # التحقق من صحة مجلد الوجهة
        if not dest_dir or not os.path.isdir(dest_dir):
            messagebox.showerror("خطأ", "يرجى اختيار مجلد الوجهة الصحيح")
            return

        # تعطيل زر البدء لمنع الضغط عليه مرة أخرى أثناء المعالجة
        self.start_button.config(state=tk.DISABLED)
        
        # بدء خيط منفصل للتصنيف لمنع تجميد واجهة المستخدم
        threading.Thread(target=self.classify_thread, args=(source_dir, dest_dir)).start()

    def classify_thread(self, source_dir, dest_dir):
        """
        خيط منفصل لعملية تصنيف جميع الصور
        
        المعلمات:
            source_dir: مسار مجلد الصور المصدر
            dest_dir: مسار مجلد الوجهة
        """
        # امتدادات الصور المدعومة
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        # الحصول على قائمة ملفات الصور في المجلد المصدر
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_exts)]
        total = len(image_files)

        # التحقق من وجود صور
        if total == 0:
            messagebox.showinfo("معلومة", "لم يتم العثور على صور")
            self.start_button.config(state=tk.NORMAL)
            return

        # إضافة رسالة إلى السجل بعدد الصور التي تم العثور عليها
        self.log_message(f"تم العثور على {total} صورة", highlight=True)

        # إنشاء المجلدات الفرعية للفئات في مجلد الوجهة
        for cat in self.categories:
            os.makedirs(os.path.join(dest_dir, cat), exist_ok=True)

        # معالجة كل صورة على حدة
        for i, filename in enumerate(image_files):
            path = os.path.join(source_dir, filename)
            
            # تصنيف الصورة
            category, prob = self.classify_image(path)
            
            # نسخ الصورة إلى المجلد المناسب
            shutil.copy2(path, os.path.join(dest_dir, category, filename))
            
            # تحديث شريط التقدم
            self.progress_var.set((i+1)/total * 100)
            
            # تحديث حالة التصنيف
            self.update_status(f"تم تصنيف: {filename}")
            
            # إضافة معلومات التصنيف إلى السجل
            # نستخدم الإبراز للفئة ونسبة الثقة
            self.log_message(f"{filename} => {category} ({prob:.2%})")

        # إكمال العملية
        self.update_status("تم الانتهاء من التصنيف")
        self.log_message("تم تصنيف جميع الصور بنجاح.", highlight=True)
        messagebox.showinfo("تم", "انتهى تصنيف الصور بنجاح!")
        
        # إعادة تفعيل زر البدء
        self.start_button.config(state=tk.NORMAL)

# نقطة الدخول للتطبيق
if __name__ == '__main__':
    # إنشاء نافذة Tkinter الرئيسية
    root = tk.Tk()
    # إنشاء نسخة من التطبيق
    app = ImageClassifierApp(root)
    # بدء دورة الأحداث
    root.mainloop()

import customtkinter as ctk
import tkinter.messagebox as messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import re

# ------------------ MODEL TRAINING ------------------
def train_model():
    df = pd.read_csv("data.csv")  # should contain text + label (for both emails and sms)
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if not os.path.exists("model.pkl"):
    train_model()

def predict_text(text):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

# ------------------ VALIDATORS ------------------
def validate_phone(phone):
    """Simple phone number suspicious check"""
    if not re.match(r"^\+?\d{7,15}$", phone):
        return "INVALID"
    suspicious_prefixes = ["+234", "+880", "+91"]  # example suspicious codes
    if any(phone.startswith(p) for p in suspicious_prefixes):
        return "SUSPICIOUS"
    return "SAFE"

# ------------------ GUI FUNCTIONS ------------------
def check_email():
    text = email_input.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter email content.")
        return
    result = predict_text(text)
    messagebox.showinfo("Result", f"Email classified as: {result.upper()}")

def check_phone():
    number = phone_input.get().strip()
    if not number:
        messagebox.showwarning("Warning", "Please enter phone number.")
        return
    result = validate_phone(number)
    messagebox.showinfo("Result", f"Phone Number Status: {result}")

def check_message():
    text = msg_input.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter message content.")
        return
    result = predict_text(text)
    messagebox.showinfo("Result", f"Message classified as: {result.upper()}")

# ------------------ UI SETUP ------------------
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

window = ctk.CTk()
window.title("Fraud & Spam Detector")
window.geometry("750x550")

tabview = ctk.CTkTabview(window, width=700, height=450)
tabview.pack(pady=20)

# ------------------ EMAIL TAB ------------------
tabview.add("📧 Email")
email_input = ctk.CTkTextbox(tabview.tab("📧 Email"), height=200, width=600, font=("Arial", 14))
email_input.pack(pady=10)
ctk.CTkButton(tabview.tab("📧 Email"), text="Check Email", command=check_email, fg_color="#007acc").pack(pady=10)

# ------------------ PHONE TAB ------------------
tabview.add("📱 Phone Number")
phone_input = ctk.CTkEntry(tabview.tab("📱 Phone Number"), width=300, placeholder_text="Enter phone number")
phone_input.pack(pady=20)
ctk.CTkButton(tabview.tab("📱 Phone Number"), text="Check Phone", command=check_phone, fg_color="#27ae60").pack(pady=10)

# ------------------ MESSAGE TAB ------------------
tabview.add("💬 Message")
msg_input = ctk.CTkTextbox(tabview.tab("💬 Message"), height=200, width=600, font=("Arial", 14))
msg_input.pack(pady=10)
ctk.CTkButton(tabview.tab("💬 Message"), text="Check Message", command=check_message, fg_color="#e67e22").pack(pady=10)

window.mainloop()

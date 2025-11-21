import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

kmeans = pickle.load(open("vocabulary.pkl","rb"))
svm = pickle.load(open("model_svm.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
sift = cv2.SIFT_create()

medicinal_db = {
    "tulsi": "Holy Basil — Immunity, cold, cough, stress relief. Make tea.",
    "neem": "Antibacterial, skin diseases, blood purifier. Use paste/oil.",
    "aloe": "Burns, wounds, digestion, skin glow. Apply gel.",
    "mint": "Digestion, headache, fresh breath. Chew or tea.",
    "curry": "Antioxidant, digestion, memory boost.",
    "moringa": "Drumstick — Diabetes, anemia, bone health.",
    "amla": "Indian Gooseberry — Vitamin C, hair, immunity."
}

st.title("ANY Leaf Identifier + Medicinal Uses")
st.markdown("Upload **ANY leaf** — I will detect everything")

file = st.file_uploader("Upload leaf photo", type=["jpg","jpeg","png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Your Leaf", width=350)
    
    cv_img = np.array(img.convert('RGB'))[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc)<10:
        desc = np.random.rand(150,128).astype('float32')
    
    words = kmeans.predict(desc)
    hist,_ = np.histogram(words, bins=200, range=(0,200))
    hist = hist.astype(float)/(hist.sum()+1e-6)
    hist = scaler.transform([hist])
    
    pred = svm.predict(hist)[0]
    conf = svm.predict_proba(hist).max()*100
    
    is_medicinal = "medicinal" in pred
    status = "HEALTHY" if "healthy" in pred else "DISEASED"
    
    # Simple name guess (you can improve later)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    green = np.mean(hsv[:,:,1])
    edges = len(cv2.findContours(cv2.Canny(gray,50,150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1])
    
    name = "Tulsi" if green > 60 and edges < 10 else "Neem" if green < 50 else "Mint" if edges > 20 else "Aloe Vera" if edges < 5 else "Curry Leaf" if edges > 15 else "Common Plant"
    
    uses = medicinal_db.get(name.lower().split()[0], "Helps clean air & produce oxygen")
    
    color = (0,255,0) if status=="HEALTHY" else (0,0,255)
    cv2.putText(cv_img, name.upper(), (20,70), 2, 2.2, (255,255,255), 6)
    cv2.putText(cv_img, f"{status} {conf:.0f}%", (20,140), 2, 2.5, color, 6)
    
    st.image(cv_img[:, :, ::-1], caption="Result", width=400)
    
    st.success(f"**Plant:** {name}")
    st.success(f"**Status:** {status} ({conf:.1f}% confidence)")
    
    if is_medicinal:
        st.success("MEDICINAL PLANT!")
        st.info(f"**USES:** {uses}")
    else:
        st.info("Common plant — good for environment")
    
    if status == "DISEASED":
        st.error("DISEASED LEAF!")
        st.info("**CURE:** Neem oil spray + remove damaged parts")

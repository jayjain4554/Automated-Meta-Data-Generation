# 📄 Automated Meta Data Generator

A lightweight, production-grade NLP-based system to **extract contextual metadata, keywords, topics, named entities, sentiment, and summaries** from documents like PDF, DOCX, and TXT.

Built using **FastAPI**, **transformers**, **BERTopic**, **KeyBERT**, and deployed with **Streamlit** UI.

---

## 🚀 Features

- 📁 Supports **PDF, DOCX, and TXT** files
- 🧠 Generates:
  - Summarized text
  - Sentiment analysis
  - Named entities grouped by type
  - Keywords (TF-IDF + KeyBERT)
  - Document structure insights (headers, paragraphs, sentence stats)
  - Topic modeling with BERTopic
- 💡 NLP stack includes HuggingFace Transformers, spaCy, KeyBERT, and more
- 🌐 Interactive Streamlit front-end
- ✅ JSON download support

---

## 📦 Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **NLP Libraries:** transformers, spacy, keybert, bertopic, pytesseract
- **PDF/Text Processing:** pdfminer.six, docx2txt, python-docx, pdfplumber

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/jayjain4554/Automated-Meta-Data-Generation.git
cd Automated-Meta-Data-Generation
````

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

## 🧪 Running the Application

### ➤ Start FastAPI backend

```bash
uvicorn app.main:app --reload
```

Runs on: [http://localhost:8000](http://localhost:8000)

### ➤ Start Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Runs on: [http://localhost:8501](http://localhost:8501)

---

## 🧾 File Support

| File Type | Supported |
| --------- | --------- |
| `.pdf`    | ✅         |
| `.docx`   | ✅         |
| `.txt`    | ✅         |

---

## 📤 Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New app”** and select the repo
4. Set the main file to `streamlit_app.py`
5. Set Python version (e.g., 3.10)
6. Add `requirements.txt` in app setup

**Make sure your repo contains only essential files, and not `venv/` or large models**

---

## 📷 Demo Screenshot

![Screenshot 2025-06-25 132613](https://github.com/user-attachments/assets/b55842de-8e5d-44b9-9fa8-d8f43b29d6ec)


---

## 🧑‍💻 Author

**Jay Jain**
BTech Chemical Engineering, IIT Roorkee

---

## 📄 License

MIT License – feel free to use and modify for academic or commercial purposes.

```

---

Let me know if you want to include a GIF/preview or auto-deploy badge in this too.
```

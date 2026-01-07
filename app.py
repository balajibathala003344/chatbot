import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# ===================== NEW IMPORTS (ADDED ONLY) =====================
import io
import textwrap   # ✅ ADDED for wrapping fix
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
# ===================================================================

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🎓 College Assistant Chatbot",
    layout="wide"
)

DATA_DIR = "data"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.npy"

# ---------------- GEMINI ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ===== GLOBAL FIX ===== */
html, body, [class*="css"]  {
    color: #EAEAEA !important;
}

/* ===== USER MESSAGE ===== */
.chat-user {
    background: #DCF8C6;
    color: #000000;
    padding: 14px;
    border-radius: 14px;
    margin: 8px 0;
    font-weight: 500;
}

/* ===== BOT MESSAGE ===== */
.chat-bot {
    background: #1E1E1E;
    color: #FFFFFF;
    padding: 14px;
    border-radius: 14px;
    margin: 8px 0;
    border-left: 5px solid #4CAF50;
}

/* ===== INFO CARD ===== */
.card {
    background: #111827;
    color: #E5E7EB;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    margin-bottom: 15px;
    font-size: 16px;
}

/* ===== INPUT BOX ===== */
input {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #334155 !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* ===== FOOTER ===== */
footer {
    text-align: center;
    margin-top: 30px;
    color: #94A3B8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- HELPERS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss():
    index = faiss.read_index(INDEX_FILE)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
    return index, chunks

def gemini_answer(prompt):
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return res.text

def rag_answer(question):
    index, chunks = load_faiss()
    embedder = load_embedder()

    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k=3)

    context = ""
    for idx in I[0]:
        context += chunks[idx]["text"] + "\n"

    if len(context.strip()) < 50:
        return None

    prompt = f"""
Answer using ONLY the context below.
If not found, say: Information not available in documents.

Context:
{context}

Question:
{question}
"""
    return gemini_answer(prompt)

def summarize_pdf(file):
    reader = PdfReader(file)
    text = " ".join([p.extract_text() or "" for p in reader.pages])

    prompt = f"""
Summarize the following document clearly in bullet points:

{text[:8000]}
"""
    return gemini_answer(prompt)

# ===================== PDF GENERATOR (FIXED + COLORED, ONLY ADDITIONS) =====================
def generate_chat_pdf(chat_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    now = datetime.now().strftime("%d-%m-%Y  %I:%M %p")

    header_color = HexColor("#2563EB")
    q_bg = HexColor("#DCF8C6")
    a_bg = HexColor("#F1F5F9")
    footer_color = HexColor("#0F172A")

    # Header bar
    c.setFillColor(header_color)
    c.rect(0, height - 90, width, 90, fill=1)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 55, "College Assistant Chat History")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, height - 75, f"Generated on: {now}")

    # Border
    c.setLineWidth(2)
    c.setStrokeColor(header_color)
    c.rect(20, 20, width - 40, height - 40)

    y = height - 120

    for chat in reversed(chat_data):

        # Question box
        c.setFillColor(q_bg)
        c.roundRect(40, y - 35, width - 80, 30, 8, fill=1)
        c.setFillColor(HexColor("#000000"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y - 22, "Q:")
        c.setFont("Helvetica", 11)
        c.drawString(70, y - 22, chat["user"])
        y -= 50

        # -------- TEXT WRAP FIX (ADDED) --------
        wrapped_lines = []
        for para in chat["bot"].split("\n"):
            wrapped_lines.extend(textwrap.wrap(para, 90) or [""])
        # --------------------------------------

        box_height = max(40, 14 * len(wrapped_lines) + 20)

        # Answer box
        c.setFillColor(a_bg)
        c.roundRect(40, y - box_height, width - 80, box_height, 8, fill=1)

        c.setFillColor(HexColor("#020617"))
        text_obj = c.beginText(50, y - 20)
        text_obj.setFont("Helvetica", 11)
        for line in wrapped_lines:
            text_obj.textLine(line)
        c.drawText(text_obj)
        y -= box_height + 25

        if y < 120:
            c.showPage()
            c.setFillColor(header_color)
            c.rect(0, height - 90, width, 90, fill=1)
            c.setFillColor(HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(width / 2, height - 55, "College Assistant Chat History")
            y = height - 120

    # Footer bar
    c.setFillColor(footer_color)
    c.rect(0, 0, width, 30, fill=1)
    c.setFillColor(HexColor("#E5E7EB"))
    c.setFont("Helvetica", 9)
    c.drawCentredString(
        width / 2,
        10,
        "Powered by Gemini + FAISS | Developed by Bathala Balaji"
    )

    c.save()
    buffer.seek(0)
    return buffer
# ===================================================================

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Options")

    gemini_mode = st.checkbox("🌐 Ask Gemini (Outside PDFs)")
    summary_mode = st.checkbox("📄 PDF Summary Mode")

    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if st.session_state.chat:
        pdf_buffer = generate_chat_pdf(st.session_state.chat)
        st.download_button(
            label="📥 Download Chat as PDF",
            data=pdf_buffer,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )

# ---------------- MAIN ----------------
st.markdown("<h1>🎓 College Assistant Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='card'>Ask questions from college PDFs or general knowledge.</div>", unsafe_allow_html=True)

question = st.text_input("💬 Enter your question")

if question:
    answer = None

    if summary_mode and uploaded:
        answer = summarize_pdf(uploaded)
    else:
        if not gemini_mode:
            answer = rag_answer(question)

        if answer is None:
            answer = gemini_answer(question)

    st.session_state.chat.insert(0, {
        "user": question,
        "bot": answer
    })

# ---------------- CHAT (LATEST ON TOP) ----------------
for chat in st.session_state.chat:
    st.markdown(f"<div class='chat-user'>👤 {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bot'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<footer>
🚀 Developed by <b>BATHALA BALAJI</b> 💻🔥
</footer>
""", unsafe_allow_html=True)

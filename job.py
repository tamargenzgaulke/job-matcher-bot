# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

!pip install transformers
!pip install pdfplumber
!pip install streamlit
!pip install torch torchvision
!pip install tensorflow




from transformers import pipeline
import pdfplumber
import streamlit as st

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# Configuração do modelo
classifier = pipeline("text-classification", model="bert-base-uncased")

# Interface Streamlit
st.title("NLP-Powered Resume Classifier")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:", text[:500])  # Exibir um trecho do texto
    results = classifier(text)
    st.write("Job Category:", results[0]['label'])
    #versao 2
    #versao 3
    
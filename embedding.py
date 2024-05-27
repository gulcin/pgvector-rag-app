# importing all the required modules
import PyPDF2
import torch
from transformers import pipeline

def generate_embeddings(tokenizer, model, device, text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return text, outputs.hidden_states[-1].mean(dim=1).tolist()


def read_pdf_file(pdf_path):
    pdf_document = PyPDF2.PdfReader(pdf_path)

    lines = []
    for page_number in  range(len(pdf_document.pages)):
        page = pdf_document.pages[page_number]

        text = page.extract_text()

        lines.extend(text.splitlines())

    return lines

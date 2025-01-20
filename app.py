import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from huggingface_hub import login
import torch

# Get the Hugging Face token from Streamlit secrets
hf_token = st.secrets["HF_TOKEN"]

# Log in with the Hugging Face token
login(token=hf_token)

# Load model and tokenizer from Hugging Face Hub
model_name = "Alaaeldin/MS-BiomedNLP-BiomedBERT-OTC"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the Streamlit interface
st.title("Biomedical Q&A with BiomedBERT")
st.markdown("Ask questions about biomedical topics based on scientific literature!")

# Get user input (question and context)
context = st.text_area("Enter the context (scientific text):")
question = st.text_input("Enter your question:")

if context and question:
    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

    # Get model's predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get answer start and end positions
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Convert token indices back to text
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    # Display the answer
    st.write(f"**Answer:** {answer}")

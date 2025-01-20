import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from huggingface_hub import login
import torch
# Login to Hugging Face
hf_token = st.secrets["HF_TOKEN"]
login(token=hf_token)
# Load the pre-trained model and tokenizer
model_name = "Alaaeldin/MS-BiomedNLP-BiomedBERT-OTC"  # Use your model name here

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Streamlit interface
st.title("BiomedBERT Q&A")

# Get user input for the question
question = st.text_input("Ask a question related to biomedical research:")

# If the user submits a question
if question:
    # Prepare input text for the model
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        # Generate the model's answer
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))

    # Display the answer
    st.write("Answer: ", answer)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Set page configuration
st.set_page_config(
    page_title="Biomedical Q&A System",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer using authentication"""
    try:
        # Access the token from Streamlit secrets
        hf_token = st.secrets["hf_token"]
        
        # Initialize tokenizer and model with authentication
        tokenizer = AutoTokenizer.from_pretrained(
            "Alaaeldin/MS-BiomedNLP-BioBERT-OTC",
            use_auth_token=hf_token
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            "Alaaeldin/MS-BiomedNLP-BioBERT-OTC",
            use_auth_token=hf_token
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error accessing token or loading model: {str(e)}")
        st.stop()

def get_answer(question, context, tokenizer, model):
    """Get answer for the question using the provided context"""
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k not in ['overflow_to_sample_mapping', 'offset_mapping']})
    
    # Get start and end positions
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Find the tokens with the highest probability for start and end positions
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    # Convert token positions to characters positions
    offset_mapping = inputs['offset_mapping'][0]
    answer_start = offset_mapping[start_index][0]
    answer_end = offset_mapping[end_index][1]
    
    # Extract the answer from the context
    answer = context[answer_start:answer_end]
    
    return answer

# Main UI
st.title("Biomedical Question Answering System")
st.markdown("This app uses the MS-BiomedNLP-BioBERT-OTC model for biomedical question answering.")

# Load model
try:
    with st.spinner("Loading model... This might take a minute."):
        tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input fields
st.subheader("Input")
context = st.text_area(
    "Context (Paste your biomedical text here):",
    height=200,
    help="Paste the biomedical text that contains the information to answer the question."
)

question = st.text_input(
    "Question:",
    help="Enter your question about the text above."
)

# Get answer when button is clicked
if st.button("Get Answer") and context and question:
    try:
        with st.spinner("Finding answer..."):
            answer = get_answer(question, context, tokenizer, model)
        
        st.subheader("Answer")
        st.write(answer)
        
        # Show confidence metrics
        st.subheader("Answer Details")
        st.info(f"Length of extracted answer: {len(answer)} characters")
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
elif st.button("Get Answer"):
    st.warning("Please provide both context and question.")

# Add footer
st.markdown("---")
st.markdown("Created with Streamlit and Hugging Face Transformers")

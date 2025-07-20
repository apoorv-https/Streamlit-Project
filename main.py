import streamlit as st
from langchain_groq import ChatGroq
import PyPDF2
import io
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq LLM
@st.cache_resource
def initialize_llm():
    models = ["gemma2-9b-it", "deepseek-r1-distill-llama-70b"]
    
    for model in models:
        try:
            return ChatGroq(
                model=model,
                temperature=0,
                groq_api_key=os.getenv("API_KEY")
            )
        except Exception as e:
            if "rate_limit" in str(e).lower():
                continue
            else:
                st.error(f"Failed to initialize {model}: {str(e)}")
                continue
    
    st.error("All models failed or hit rate limits")
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def generate_response(llm, text, option, topic, number):
    """Generate summary or questions"""
    try:
        if option == "Summary":
            if topic.strip():
                prompt = f"Provide a detailed summary of this document focusing on {topic}:\n\n{text}"
            else:
                prompt = f"Provide a detailed summary of this document:\n\n{text}"
        
        else:  # Questions
            if topic.strip():
                prompt = f"Create {number} educational questions about {topic} based on this document. Format as Q1., Q2., etc.:\n\n{text}"
            else:
                prompt = f"Create {number} educational questions based on this document. Format as Q1., Q2., etc.:\n\n{text}"
        
        response = llm.invoke(prompt)
        content = response.content
        
        if option == "Summary":
            # Minimal filtering for summary - just remove empty lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            return '\n\n'.join(lines)
        
        else:  
            
            return content
    
    except Exception as e:
        st.error(f"Error generating {option.lower()}: {str(e)}")
        return None

def main():
    st.title("📄 PDF Assistant")
    
    # UI Elements
    option = st.selectbox("Choose what you want:", ["Summary", "Questions"])
    topic = st.text_input("Enter topic (optional):", placeholder="Leave blank for general processing")
    
    if option == "Questions":
        number = st.slider("Number of questions:", 1, 20, 5)
    else:
        number = 5
    
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    # File size check
    if uploaded_file and uploaded_file.size > 10 * 1024 * 1024:
        st.error(" File too large! ")
        return
    
   
    if uploaded_file:
     
        llm = initialize_llm()
        if not llm:
            return
        
        
        with st.spinner("Reading PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if not text:
            st.error(" Please check if it's a valid PDF with readable text.")
            return
        
        if len(text) < 50:
            st.warning("Very little text found in PDF. Results may be limited.")
        
      
        with st.spinner(f"Generating {option.lower()}..."):
            # Limit text to avoid token limits
            limited_text = text[:8000] if len(text) > 8000 else text
            result = generate_response(llm, limited_text, option, topic, number)
        
        if result:
            
            st.write(result)
        else:
            st.error(f" Failed to generate {option.lower()}. Please try again.")

if __name__ == "__main__":
    main()
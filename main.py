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
    try:
        return ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            groq_api_key=os.getenv("API_KEY")
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
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
        
        else:  # Questions
            # Simple question formatting
            lines = content.split('\n')
            questions = []
            question_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if it's already a question or contains question content
                if (line[0].isdigit() or line.startswith('Q') or 
                    line.endswith('?') or any(word in line.lower() for word in ['what', 'how', 'why', 'when', 'where'])):
                    
                    question_count += 1
                    # Remove existing numbering and add consistent format
                    clean_line = line
                    if line[0].isdigit():
                        clean_line = line.split('.', 1)[-1].strip()
                    if line.startswith('Q'):
                        clean_line = line.split('.', 1)[-1].strip()
                    
                    questions.append(f"Q{question_count}. {clean_line}")
                    
                    if question_count >= number:
                        break
            
            return '\n\n'.join(questions) if questions else content
    
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
        st.error("⚠️ File too large! ")
        return
    
    # Main processing
    if uploaded_file:
        # Initialize LLM
        llm = initialize_llm()
        if not llm:
            return
        
        # Extract text
        with st.spinner("Reading PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if not text:
            st.error("❌ Could not extract text from PDF. Please check if it's a valid PDF with readable text.")
            return
        
        if len(text) < 50:
            st.warning("⚠️ Very little text found in PDF. Results may be limited.")
        
        # Generate response
        with st.spinner(f"Generating {option.lower()}..."):
            # Limit text to avoid token limits
            limited_text = text[:8000] if len(text) > 8000 else text
            result = generate_response(llm, limited_text, option, topic, number)
        
        if result:
            
            st.write(result)
        else:
            st.error(f"❌ Failed to generate {option.lower()}. Please try again.")

if __name__ == "__main__":
    main()
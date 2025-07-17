import streamlit as st
from langchain-groq import ChatGroq
import PyPDF2
import io
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq LLM
@st.cache_resource
def initialize_llm():
  return ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        groq_api_key=os.getenv("API_KEY")
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_bytes = pdf_file.read()
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def generate_response(llm, text, option, topic, number):
    """Generate summary or questions based on option with aggressive filtering"""
    
    import re

    if option == "Summary":
        prompt = f"""You are a helpful assistant. Please provide a **concise and coherent summary** of the document below. Focus on the main points and keep it in **paragraph form**. 
        **Topic:** {topic} 
        **Document:** {text} 
        **Summary:**"""
    
        response = llm.invoke(prompt)
        content = response.content
        
        # Smart filtering that considers context
        lines = content.split('\n')
        summary_lines = []
        
        # Keywords that indicate important content vs. LLM artifacts
        important_keywords = ['key', 'important', 'main', 'significant', 'primary', 'crucial', 'essential', 'findings', 'results', 'conclusion']
        llm_artifacts = ['here are', 'let me', 'i will', 'to summarize', 'in summary', 'the following']
    
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are clearly questions
            if any(indicator in line.lower() for indicator in ['question', 'q:']):
                continue
            
            # Skip lines ending with question marks
            if line.endswith('?'):
                continue
            
            # Check if line is a list item
            if re.match(r'^(\d+\.|[a-z]\.|[ivxlc]+\.|[-•*])', line):
                # Extract content without list formatting
                content_without_formatting = re.sub(r'^(\d+\.|[a-z]\.|[ivxlc]+\.|[-•*])\s*', '', line)
                
                # Keep if it contains important keywords or substantial content
                if (any(keyword in content_without_formatting.lower() for keyword in important_keywords) or 
                    len(content_without_formatting) > 20):
                    summary_lines.append(content_without_formatting)
            
            # Skip LLM artifacts (meta-commentary)
            elif any(artifact in line.lower() for artifact in llm_artifacts):
                continue
            
            # Keep substantial paragraph content
            elif len(line) > 15:  # Filter out very short lines that might be artifacts
                summary_lines.append(line)
        
        # ✅ FIXED: Return statement is now OUTSIDE the for loop
        return '\n\n'.join(summary_lines)
        
    else:  # Questions
        question_placeholders = "\n".join([f"{i+1}." for i in range(number)])

        prompt = f"""You are an expert assistant that creates **{number} educational questions**.

        Given the topic and content below, generate **{number} numbered and relevant questions** that test understanding of the text.

        **Topic:** {topic}

        **Document:**
        {text}

        **Questions:**
        {question_placeholders}
        """
        response = llm.invoke(prompt)
        content = response.content
        
        # Extract only numbered questions
        lines = content.split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions (dynamic range)
            if any(line.startswith(f"{i}.") for i in range(1, number + 1)):
                questions.append(line)
        
        # If we didn't get expected number of questions, return original content
        if len(questions) < number:
            return content
        
        return '\n\n'.join(questions[:number])  # Return requested number of questions


def main():
    st.title("📄 PDF Assistant")
    # Simple selectbox
    option = st.selectbox("Choose what you want:", ["Summary", "Questions"])
    topic = st.text_input("Enter the topic")
    
    if option == "Questions":
        number = st.slider("Select a number", 1, 50, 5)
    else:
        number = 5  # Default for summary
    
    # File uploader
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file and topic:  # ✅ FIXED: Added topic validation
        try:
            # Extract text
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if text.strip():
                # Generate response
                llm = initialize_llm()
                with st.spinner(f"Generating {option.lower()}..."):
                    result = generate_response(llm, text[:15000], option, topic, number)

                st.write(result)
            else:
                st.error("No text found in PDF")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    elif uploaded_file and not topic:
        st.warning("Please enter a topic to proceed")

if __name__ == "__main__":
    main()
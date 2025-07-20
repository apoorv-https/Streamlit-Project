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
        if topic.strip():  # If topic is provided
            prompt = f"""You are a helpful assistant. Please provide a **concise and coherent summary** of the document below. Focus on the main points and keep it in **paragraph form**. 
            
            **Topic:** {topic} 
            **Document:** {text} 
            **Summary:**"""
        else:  # If no topic provided, make it generalized
            prompt = f"""You are a helpful assistant. Please provide a **concise and coherent summary** of the document below. Focus on the main points and key information and keep it in **paragraph form**. 
            
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
        
     
        return '\n\n'.join(summary_lines)
        
    else:  # Questions
        question_placeholders = "\n".join([f"{i+1}." for i in range(number)])

        if topic.strip():  # If topic is provided
            prompt = f"""You are an expert assistant that creates educational questions. 

            Please READ the document content below and CREATE {number} original questions that test understanding of the material related to the topic "{topic}".

            Generate completely NEW questions based on what you learned from reading the document. DO NOT extract existing questions from the text.

            **Document Content:**
            {text}

            Please generate exactly {number} questions in this format:
            1. [Your first question here]
            2. [Your second question here]
            And so on...

            Focus on creating questions that test comprehension, analysis, and application of the concepts from the document."""
        else:  # If no topic provided, make it generalized
            prompt = f"""You are an expert assistant that creates educational questions.

            Please READ the document content below and CREATE {number} original questions that test understanding of the key concepts and information.

            Generate completely NEW questions based on what you learned from reading the document. DO NOT extract existing questions from the text.

            **Document Content:**
            {text}

            Please generate exactly {number} questions in this format:
            1. [Your first question here]  
            2. [Your second question here]
            And so on...

            Focus on creating questions that test comprehension, analysis, and application of the concepts from the document."""
            
        response = llm.invoke(prompt)
        content = response.content
        
        # Extract and format numbered questions
        lines = content.split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Look for numbered questions (more flexible patterns)
            # Matches: "1.", "1)", "Q1.", "Question 1:", etc.
            if (re.match(r'^(\d+\.|\d+\)|Q\d+\.|Question\s+\d+:)', line) or
                any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, number + 1))):
                
                # Clean up the question format to ensure consistent "Q1.", "Q2." format
                question_text = re.sub(r'^(\d+\.|\d+\)|Q\d+\.|Question\s+\d+:)\s*', '', line)
                question_number = len(questions) + 1
                formatted_question = f"Q{question_number}. {question_text}"
                questions.append(formatted_question)
            
            # Also catch questions that might not be perfectly numbered but contain question content
            elif line.endswith('?') and len(line) > 10 and len(questions) < number:
                question_number = len(questions) + 1
                formatted_question = f"Q{question_number}. {line}"
                questions.append(formatted_question)
        
        # If we got the expected number of questions, return them
        if len(questions) >= number:
            return '\n\n'.join(questions[:number])
        
        # If not enough questions found, try a different approach - look for any question-like content
        all_lines = [line.strip() for line in lines if line.strip()]
        potential_questions = []
        
        for line in all_lines:
            # Look for any line that could be a question
            if (len(line) > 15 and 
                (line.endswith('?') or 
                 any(word in line.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']) or
                 re.match(r'^\d+', line))):
                potential_questions.append(line)
        
        # Format the potential questions
        formatted_questions = []
        for i, question in enumerate(potential_questions[:number]):
            # Remove any existing numbering
            clean_question = re.sub(r'^(\d+\.|\d+\)|Q\d+\.|Question\s+\d+:)\s*', '', question)
            formatted_questions.append(f"Q{i+1}. {clean_question}")
        
        if len(formatted_questions) >= number:
            return '\n\n'.join(formatted_questions[:number])
        
        # Last resort - return original content if nothing worked
        return content


def main():
    
    st.title("📄 PDF Assistant")
    # Simple selectbox
    option = st.selectbox("Choose what you want:", ["Summary", "Questions"])
    topic = st.text_input("Enter the topic (optional - leave blank for generalize)")
    
    if option == "Questions":
        number = st.slider("Select a number", 1, 50, 5)
    else:
        number = 5  

    
    # File uploader
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
      
   
    
    if uploaded_file:
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error(" File too large!")
            st.stop()
    
    if uploaded_file:  # Remove the topic requirement
        try:
            # Extract text
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if text.strip():
                # Generate response
                llm = initialize_llm()
                with st.spinner(f"Generating {option.lower()}..."):
                    result = generate_response(llm, text[:6000], option, topic if topic else "", number)

                st.write(result)
            else:
                st.error("No text found in PDF")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
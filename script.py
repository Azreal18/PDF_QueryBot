import warnings
warnings.filterwarnings("ignore")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import os 
import re 
import pandas as pd
import streamlit as st
import PyPDF2 
import random
load_dotenv()


def get_pdf_data(pdf_file_path):
  """
  Extracts text and metadata from a PDF file.

  Args:
    pdf_file_path: The path to the PDF file.

  Returns:
    A dictionary containing the extracted data, including:
      text: The complete text content of the PDF.
      metadata: A dictionary of the PDF metadata (titles, author, etc.).
  """

  # Open the PDF file
  pdf_reader = PyPDF2.PdfReader(pdf_file_path)

  # Extract text data
  all_page_text = ""
  for page in pdf_reader.pages:
    page_text = page.extract_text()
    all_page_text += page_text
    
    return all_page_text


def clean_up(input_text):
    # Remove noisy characters except for letters, digits, whitespace, ? and !
    cleaned_text = re.sub(r'[^\w\s?!,\-%\(\)]', '', input_text)

    # Remove extra new lines
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

    return cleaned_text.strip()

def file_processing(file_path):
    """
    This function processes a file to generate document-based questions and answers.

    Args:
        file_path (str): The path to the file.

    Returns:
        tuple: A tuple containing the document-based questions and answers.
    """

    # Load data from PDF
    loader = get_pdf_data(file_path)
    page_content = loader

    page_content = clean_up(page_content)
        
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
    )

    chunks_ques = splitter_ques_gen.split_text(page_content)
    print("Number of chunks: ", len(chunks_ques))

    ques_gen = [Document(page_content=t) for t in chunks_ques]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 100,
        length_function = len,
    )

    answer_gen = splitter_ans_gen.split_documents(
        ques_gen
    )
    print("Number of chunks: ", len(answer_gen))
    return ques_gen, answer_gen


def pipeline(file_path):
    """
    This function represents the pipeline for generating questions and answers based on a given file.

    Args:
        file_path (str): The path to the file.

    Returns:
        tuple: A tuple containing the answer generation chain and the filtered question list.
    """

    # Process the file to generate document-based questions and answers
    print("Processing the file...")
    ques_gen, answer_gen = file_processing(file_path)

    # Initialize the LLM question generation pipeline
    ques_gen_pipeline = llm_ques

    # Prompt template for generating questions
    base_template = """
    As an expert in question generation, your task is to create insightful questions based on the given input.
    Your goal is to help individuals gain a deeper understanding of the topic and encourage critical thinking.
    Please generate questions based on the following input:

    ------------

    {text}

    ------------

    Your questions should explore different aspects of the input and test the knowledge base.
    Ensure that no important information is overlooked and avoid repeating questions.
    Feel free to ask factual and explanatory questions.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=base_template, input_variables=["text"])

    # Template for refining existing questions
    refined_template = ("""
    As an expert in question generation, your task is to refine the existing questions based on the given context.
    Your goal is to help individuals prepare for a knowledge test.
    We have received some practice questions to a certain extent: {existing_answer}.
    Now, we have additional context to consider:
    ------------
    {text}
    ------------

    Refine the original questions in English based on the new context.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refined_template,
    )

    # Load the LLM question generation chain
    print("Loading the LLM question generation chain...")
    ques_gen_chain = load_summarize_chain(llm=ques_gen_pipeline, 
                                          chain_type="refine", 
                                          verbose=True, 
                                          question_prompt=PROMPT_QUESTIONS, 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Generate questions using the LLM question generation chain
    ques = ques_gen_chain.run(ques_gen)

    # Split the generated questions into a list
    ques_list = ques.split("\n")

    # Filter the questions to include only those ending with '?' or '.'
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.') and len(element) > 15]

    filtered_ques_list =[string.strip() for string in filtered_ques_list]
    
    # Initialize the HuggingFace  embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create a vector store using FAISS from the document-based answers
    vector_store = FAISS.from_documents(answer_gen, embeddings)

    # Initialize the LLM answer generation pipeline
    answer_gen_pipeline = llm_ans


    # Initialize the retrieval-based QA system using the LLM answer generation pipeline and the vector store
    print("Initializing the retrieval-based QA system...")
    ans_gen_chain = RetrievalQA.from_chain_type(llm=answer_gen_pipeline, 
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever(k=2))

    return ans_gen_chain, filtered_ques_list

def gen(file_path):
    # Run the question-answering pipeline and get the answer generation chain and question list
    print("Running the question-answering pipeline...")
    ans_gen_chain, ques_list = pipeline(file_path)
    
    # Create an empty dictionary to store the questions and answers
    qa_dict = {"Question": [], "Answer": []}
    selected_questions = random.sample(ques_list, 10)
    
    # Iterate through the question list

    print("Generating answers...")
    for question in selected_questions:
        print("Question: ", question)
        
        # Generate the answer using the answer generation chain
        answer = ans_gen_chain.run(question)
        print("Answer: ", answer)
        print("--------------------------------------------------\n\n")
        
        # Append the question and answer to the dictionary
        qa_dict["Question"].append(question)
        qa_dict["Answer"].append(answer)
    
    # Convert the dictionary to a DataFrame
    qa_df = pd.DataFrame(qa_dict)
    
    return qa_df
# Streamlit app
def main():
    # Set the theme
    st.set_page_config(page_title="PDF Processing App", page_icon=":books:", layout="wide", initial_sidebar_state="expanded")
    st.title("PDF Processing App")

    # Custom drag-and-drop functionality
    uploaded_file = st.file_uploader("Drag and drop a PDF file here", type=["pdf"])

    # Display the uploaded file and process the PDF file
    if uploaded_file is not None:
        st.markdown("### Uploaded PDF File")
        st.write(uploaded_file.name)
        st.write(f"File Size: {round(uploaded_file.size / 1024, 2)} KB")

        # Process the PDF file using the gen() function
        df = gen(uploaded_file)

        # Display the DataFrame
        st.markdown("### Processed Data")
        st.write(df)

        # Option to save DataFrame as CSV
        if st.button("Save as CSV"):
            csv_file = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_file,
                file_name="Results.csv",
                key="download_button_csv",
            )

        # Option to save DataFrame as JSON
        if st.button("Save as JSON"):
            json_file = df.to_json(orient="records")
            st.download_button(
                label="Download JSON",
                data=json_file,
                file_name="Results.json",
                key="download_button_json",
            )
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the language model
llm_ques = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1",model_kwargs={"temperature": 0.05, "max_new_tokens": 1048})
llm_ans = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1",model_kwargs={"temperature": 0.3, "max_new_tokens": 1048})
if __name__ == "__main__":
    main()




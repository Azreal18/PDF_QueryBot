# PDF Processing App

This is a Python application that processes PDF files to generate document-based questions and answers. It utilizes various libraries and models for text extraction, question generation, and answer generation.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Azreal18/PDF_QueryBot.git
    ```

2. **Python Version:**

- This code requires Python version 3.11.5. Please ensure you have this version installed.

3. Create a virtual environment using pipenv:

    ```bash
    pipenv install
    ```

   or using conda:

    ```bash
    conda create --name myenv
    conda activate myenv
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. **Hugging Face Model:**

- This code utilizes a model from Hugging Face.
- To access it, you'll need to create a `.env` file in the project's root directory with the following structure:

   ```
   HUGGINGFACEHUB_API_TOKEN = "YOUR_ACCESS_TOKEN"
   ```

   - Replace `YOUR_ACCESS_TOKEN` with your Hugging Face API token.

## Usage

1. Run the application:

    ```bash
    streamlit run script.py
    ```

2. Drag and drop a PDF file into the application.

3. The application will process the PDF file and generate a DataFrame containing the questions and answers.

4. The processed data can be saved as a CSV or JSON file by clicking the corresponding buttons.

## Libraries and Models Used

- `langchain`: A library for text splitting, document representation, and question generation.
- `langchain_community`: A library for embeddings and vector stores.
- `PyPDF2`: A library for extracting text and metadata from PDF files.
- `dotenv`: A library for loading environment variables from a `.env` file.
- `pandas`: A library for data manipulation and analysis.
- `streamlit`: A library for building interactive web applications.
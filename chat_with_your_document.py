# Import the supporting libraries
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="Chat with Your Document",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
log_file = "log.txt"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Verify log file creation
if os.path.exists(log_file):
    logger.info("Log file created successfully.")
else:
    logger.error("Log file was not created.")


@st.cache_resource(show_spinner=True)
def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="yourRAG")
    logger.info("Vector DB created")
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")

    # Initialize LLM
    llm = ChatOllama(model=selected_model)

    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    # UI Layout inside a box with columns
    with st.container(height= 640, border= True):

        # Get available models
        models_info = ollama.list()
        available_models = extract_model_names(models_info)

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = None

        # Separate the File and Chat
        col1, col2 = st.columns([1.5, 2])

        # Left column (Upload and Display PDF Pages)
        with col1:

            file_upload = st.file_uploader(
                "Upload a PDF file here",
                type="pdf",
                accept_multiple_files=False,
                key="pdf_uploader",
            )

            if file_upload:
                if st.session_state["vector_db"] is None:
                    progress_bar = st.progress(0)
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    progress_bar.progress(50)
                    pdf_pages = extract_all_pages_as_images(file_upload)
                    st.session_state["pdf_pages"] = pdf_pages
                    progress_bar.progress(100)

            # Display PDF pages only before clicking "Delete Collection"
            if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
                st.write("**Uploaded PDF Pages**")
                with st.container(height=300, border=True):
                    # Removed the key parameter from st.image()
                    for page_image in st.session_state["pdf_pages"]:
                        st.image(page_image, width=700)

            # Button to delete collection
            if st.button("Delete Collection", type="secondary", key="delete_button"):
                if "vector_db" in st.session_state:
                    delete_vector_db(st.session_state["vector_db"])
                    del st.session_state["vector_db"]
                if "pdf_pages" in st.session_state:
                    del st.session_state["pdf_pages"]
                st.success("Collection deleted successfully!")

        # Right column (Chat Interface)
        with col2:

            selected_model = available_models[0] # selected_model = "llama2:latest"

            # Display the selected model name on the page
            st.markdown(f"**Model:** {selected_model}")

            message_container = st.container(height=510, border=True)

            # Displaying all previous messages inside the message container
            with message_container:
                if "messages" in st.session_state:
                    for message in st.session_state["messages"]:
                        role = message["role"]
                        content = message["content"]
                        st.markdown(f"**{role.title()}:** {content}")

            if prompt := st.chat_input("Enter your prompt here ....", key="chat_input"):
                try:
                    if prompt.lower() in ["exit", "close"]:
                        st.session_state["messages"].append({"role": "assistant", "content": "Collection will be deleted and the page will be closed."})
                        delete_vector_db(st.session_state["vector_db"])
                        st.stop()
                    else:
                        # Append user message to session state
                        st.session_state["messages"].append({"role": "User", "content": prompt})
                        # Display user message in the message container
                        with message_container:
                            st.markdown(f"**User:** {prompt}")

                        # Process and display assistant response
                        with message_container:
                            with st.spinner(":green[processing...]"):
                                if st.session_state["vector_db"] is not None:
                                    response = process_question(prompt, st.session_state["vector_db"], selected_model)
                                    st.markdown(f"**Assistant:** {response}")
                                else:
                                    st.warning("Please upload a PDF file first.")

                        # Add assistant response to chat history
                        if st.session_state["vector_db"] is not None:
                            st.session_state["messages"].append({"role": "assistant", "content": response})

                except Exception as e:
                    logger.exception("An error occurred while processing the question.")
                    st.error(f"Error occurred: {str(e)}")
            else:
                if st.session_state["vector_db"] is None:
                    st.warning("Upload a PDF file to begin chat...")      

# Run the application
if __name__ == "__main__":
    main()

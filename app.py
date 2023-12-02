# Import Langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# Import Environment Modules
import os
from dotenv import load_dotenv

# Import API Modules
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Import Other Modules
import json
import logging
import warnings
warnings.filterwarnings("ignore")

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def environment_setup() -> None:
    """
    Load environment variables and set OpenAI API key.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_documents(document_path: str) -> list:
    """
    Load the pdf file and split it into pages.
    """
    try:
        loader = PyPDFLoader(document_path)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        logging.error(f"Error loading documents from {document_path}: {e}")
        return []

def split_documents(pages: list) -> list:
    """
    Split the pages into chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )
        docs = text_splitter.split_documents(pages)
        return docs
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        return []

def process_documents() -> list:
    """
    Process all documents in the specified path.
    """
    document_paths = [os.path.join(config['DOCUMENTS_PATH'], f) for f in os.listdir(config['DOCUMENTS_PATH']) if f.endswith(".pdf")]

    all_docs = []
    for document_path in document_paths:
        pages = load_documents(document_path)
        docs = split_documents(pages)
        all_docs.extend(docs)

    return all_docs

def embeddings(docs: list) -> FAISS:
    """
    Load the embeddings and store them in a vector store.
    """
    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        return db
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None

def initialize_model() -> OpenAI:
    """
    Initialize the model.
    """
    llm = OpenAI()
    return llm

def LLM_chain(llm: OpenAI, db: FAISS) -> RetrievalQA:
    """
    Create a retrieval chain with the LLM and vector store.
    """
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 5}))
    return chain

def initialize_all() -> tuple:
    """
    Initialize all components.
    """
    environment_setup()
    docs = process_documents()
    db = embeddings(docs)
    llm = initialize_model()
    llm_chain = LLM_chain(llm, db)
    return llm_chain, db

def process_message(chain: RetrievalQA, user_message: str, db: FAISS) -> str:
    """
    Process the user's message and return the bot's response.
    """
    try:
        query = user_message
        docs = db.similarity_search(query)
        result = chain.run(input_documents=docs, query=query)
        return result
    except Exception as e:
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "Sorry, I couldn't understand your message."


def setup_fastapi(llm_chain: RetrievalQA, db: FAISS) -> FastAPI:
    """
    Setup FastAPI with routes.
    """
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def read_root() -> HTMLResponse:
        """
        Serve the chatbot HTML page.
        """
        try:
            with open('templates/chatbot.html', 'r') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            logging.error(f"Error reading HTML file: {e}", exc_info=True)
            return HTMLResponse(content="Sorry, something went wrong.", status_code=500)

    @app.get("/chatbot/{user_message}")
    def get_bot_response(user_message: str) -> JSONResponse:
        """
        Process the user's message and return the bot's response.
        """
        try:
            bot_response = process_message(llm_chain, user_message, db)
            return JSONResponse(content={"answer": bot_response})
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            return JSONResponse(content={"answer": "Sorry, something went wrong."})

    return app

if __name__ == "__main__":
    try:
        llm_chain, db = initialize_all()
        fastapi_app = setup_fastapi(llm_chain, db)
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Error during initialization: {e}", exc_info=True)
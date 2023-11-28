# Langchain Document Processing

This repository contains a Python application for processing documents using the Langchain library. The application loads PDF documents, splits them into chunks, generates embeddings for each chunk, and stores the embeddings in a vector store. It also provides a FastAPI interface for interacting with the processed documents.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/yourrepository.git
```

2. Navigate to the cloned repository:

```bash
cd yourrepository
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory of the repository and add the following environment variables:

```bash
echo "OPENAI_API_KEY="yourapikey"" >> .env

echo "OPENAI_ORGANIZATION_ID="yourorganizationid"" >> .env
```

5. Create a config.json file in the root directory of the repository and add the following configuration:
```json
{
    "DOCUMENTS_PATH": "path_to_your_documents",
    
}
```

6. Run the application:

```bash
python app.py
```

Note: The application will start a FastAPI server on http://0.0.0.0:8000.


# How It Works

The application uses several components from the Langchain library:

1. PyPDFLoader: This component loads PDF documents and splits them into pages.
2. RecursiveCharacterTextSplitter: This component splits the pages into chunks.
3. OpenAIEmbeddings: This component generates embeddings for each chunk.
4. FAISS: This component stores the embeddings in a vector store.
5. RetrievalQA: This component provides a retrieval-based question answering system.
6. OpenAI: This component is a language model.

The application also uses FastAPI to provide an interface for interacting with the processed documents. It provides two endpoints:

1. /: This endpoint serves a chatbot HTML page.
2. /chatbot/{user_message}: This endpoint processes the user's message and returns the bot's response.


## Contributing

Contributions are welcome! Please read the contributing guidelines before getting started.

## License

This project is licensed under the terms of the MIT license. See the LICENSE file for details.


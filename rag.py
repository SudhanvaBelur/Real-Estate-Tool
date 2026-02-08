from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate_collection"
VECTORSTORE_PATH = Path(__file__).parent / "resources/vectorstore"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

llm=None
vector_store=None

def initialize_components():
    """
    This is a function that initializes the components required for RAG.
    """
    global llm, vector_store
    if llm is  None:   # Initialize the language model if not already initialized
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500, top_p=0.7)

    ef=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,  model_kwargs={"trust_remote_code": True})
    
    if vector_store is None:  # Initialize the vector store if not already initialized
        # Initialize Chroma vector store with the specified collection name, embedding function, and persistence directory
        vector_store= Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_PATH),
        )


def process_urls(urls):   # create a function to process the input URLs and store the data in the vector database
    """
    This is a function that scraps/extract data from urls and store it in a vector database.
    :param urls: input urls
    :return:
    """

    yield("Initializing Components...") # Initialize the language model and vector store components by calling the above function
    initialize_components()
    
    yield("Resetting Vector Store...")
    vector_store.reset_collection()

    yield("Loading data from urls")  # Load the data from the input URLs using UnstructuredURLLoader and store it in a variable called data
    # Define headers to mimic a web browser and avoid potential blocking by websites when scraping data
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    data=loader.load()  # Load the data from the URLs

    yield("Splitting texts")
    splitter= RecursiveCharacterTextSplitter(  # Split the loaded data into smaller chunks using RecursiveCharacterTextSplitter and store the resulting chunks in a variable called docs
        separators=["\n\n", "\n", ".", " "],  # Define the separators to use for splitting the text into chunks. The text will be split at double newlines, single newlines, periods, and spaces in the specified order.
        chunk_size=CHUNK_SIZE
    )
    chunks=splitter.split_documents(data)

    yield("Adding documents to vector store")  # Generate unique IDs for each chunk of text and add the chunks along with their corresponding IDs to the vector database using the add_documents method of the vector store
    uuids= [str(uuid4()) for _ in range (len(chunks))]
    vector_store.add_documents(chunks, ids=uuids)

    yield("Done adding Documents to vector database")

def generate_answer(query):  # create a function to generate an answer for the input query using the data stored in the vector database
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever())  # Create a RetrievalQAWithSourcesChain by combining the question-answering chain with 
    result = chain.invoke({"query": query}, return_only_outputs=True)

    return result["result"]


if __name__=="__main__":
    urls=[
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    process_urls(urls)
    answer= generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    print(f"Answer: {answer}")
    

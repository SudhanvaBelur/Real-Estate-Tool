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
    if llm is  None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500, top_p=0.7)

    ef=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if vector_store is None:
        vector_store= Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_PATH),
        )


def process_urls(urls):
    """
    This is a function that scraps/extract data from urls and store it in a vector database.
    :param urls: input urls
    :return:
    """

    yield("Initializing Components...")
    initialize_components()
    
    yield("Resetting Vector Store...")
    vector_store.reset_collection()

    yield("Loading data from urls")
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    data=loader.load()

    yield("Splitting texts")
    splitter= RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    chunks=splitter.split_documents(data)

    yield("Adding documents to vector store")
    uuids= [str(uuid4()) for _ in range (len(chunks))]
    vector_store.add_documents(chunks, ids=uuids)

    yield("Done adding Documents to vector database")

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever())
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
    

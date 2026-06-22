from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv



load_dotenv()

embedding_model = OllamaEmbeddings(model='nomic-embed-text')

documents = [
    Document(
        page_content='product SKU-7742X  is our flagship router. It supports gigabit speeds and advanced Qos features',
        metadata={'type':'product'}
    ),
    
    Document(
        page_content='for network connectivity issues , first check the ethernet cable and router status light',
        metadata={'type':'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server rejected the connections. check firewall settings',
        metadata={'type':'error'}
    )
]

vector_store = Chroma.from_documents(
    documents,
    embedding_model,
    collection_name='hybrid_test'
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)

print('vector retriever ready')

bm25 = BM25Retriever.from_documents(
    documents,
    k=3
)

print('BM25 retriever ready')

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.5,0.5]
)

print('ensemble retriever ready')

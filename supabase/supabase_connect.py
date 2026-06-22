from dotenv import load_dotenv
import os
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

load_dotenv()

supabase_url = os.getenv('supabase_database_url')

database_url = supabase_url 

def connect_to_supabase():
    '''connect to supabase pgvector'''

    embedding_model = OllamaEmbeddings(model='nomic-embed-text')

    vectorstore = PGVector(
        embeddings=embedding_model,
        collection_name='production_docs',
        connection=database_url,
        use_jsonb=True,
    )

    return vectorstore

def verify_connection(vectorstore):
    from langchain_core.documents import Document

    test_document = Document(
        page_content='tthis is a test document.',
        metadata={'test':True}
    )

    try:
        ids= vectorstore.add_documents([test_document])
        print("-----> added test document")

        results = vectorstore.similarity_search('test document')
        if results:
            print(f"search works: {results[0].page_content}")

        vectorstore.delete(ids)
        print('---cleaned up nicely')
        return True

    except Exception as e:
        print(f'there was an error {e}')
        return False    
    

vector_store = connect_to_supabase()

verify_connection(vector_store)
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
from langsmith import traceable
import os



load_dotenv()

os.environ['LANGSMITH_TRACING'] = 'true'

@traceable(name='basic_chaining')
def demo_basic_tracing():
    '''basic LangSmith tracing'''

    llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.2, 
    )

    prompt=ChatPromptTemplate.from_template(
        'explain {topic} in one sentence'
    )

    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        'topic':'langsmith provides observability for llm'
    })

    print(f'result:{result}')

demo_basic_tracing()    
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
import tempfile

embedding_model = OllamaEmbeddings(model='nomic-embed-text')

KNOWLEDGE_BASE ='''
Tennis is a dynamic, globally recognized racket sport that blends physical endurance, strategic agility, and mental fortitude. Evolving from a French medieval handball game into a modern phenomenon, it is celebrated for its health benefits, rich history, and the prestige of its four major global tournaments.Explore how this multifaceted sport continues to captivate millions across three key areas:The Origins and Evolution of the GameModern tennis—initially called "lawn tennis"—originated in Birmingham, England, during the late 19th century. Its roots, however, trace back centuries to the 12th-century French game jeu de paume (game of the palm). Though the sport began as an aristocratic pursuit reserved for Victorian elites, it has since evolved into an accessible, universally played game. Today, the sport's crowning achievements are the four Grand Slam tournaments: the Australian Open, the French Open, Wimbledon, and the US Open. These tournaments showcase the highest levels of athleticism and determine the sport's greatest champions.The Mechanics and Scoring SystemTennis can be played as a singles match (two players) or a doubles match (four players) on a rectangular court marked by a central net. Players use stringed rackets to strike a felt-covered hollow rubber ball into their opponent's side of the court. The scoring system is unique and highly distinct:Games: Points are tallied as 15, 30, and 40. A tie at 40-40 is called "deuce," and a player must win two consecutive points to win the game.Sets: A set is won by the first player to win six games, provided they are leading by at least two games. Tiebreaks are used if the set reaches a 6-6 deadlock.Physical and Mental BenefitsBeyond professional competition, tennis is an excellent "sport for a lifetime". Physically, the game is a full-body workout that improves cardiovascular endurance, muscle tone, and agility. Mentally, it is an intense game of chess played at high speeds. Players must constantly analyze their opponents, anticipate ball trajectories, and rapidly adjust their strategies. Mastering the sport requires immense discipline and emotional control, making it both a fantastic physical outlet and a rewarding mental exercise.

'''


def create_kb():
    '''create a vector store'''

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadata={'source':'langchain_knowledge_base.md'})
    
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store

def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':2})
    llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.2, 
    )


    prompt = ChatPromptTemplate.from_template(
    '''
    Answer the question only based on the following context.
    If the answer is not in the context, say "I don't know".
    Do not use any prior knowledge.

    Context:
    {context}

    Question: {question}
    Answer:
    '''
    )

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])
    

    rag_chain = (
        {'context': retriever | format_docs,'question': RunnablePassthrough()}
        | prompt
        | llm     
        | StrOutputParser()
    )

    questions = [
        'what is langchain?',
        'what tennis rules',
        'who invented tennis'
    ]

    print('basic RAG Demo:\n')
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f'q:{q}')
        print(f'A:{answer}\n')

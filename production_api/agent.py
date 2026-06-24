from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_postgres import PGVector
from langsmith import traceable

from config import get_settings

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    route: str
    context: str


class productionAgent:

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOllama(
            model=settings.primary_model,
            temperature=0,
            base_url="http://host.docker.internal:11434"
        )

        self.fallback_llm = ChatOllama(
            model=settings.fallback_model,
            temperature=0,
            base_url="http://host.docker.internal:11434"
        )

        # RAG setup
        embeddings = OllamaEmbeddings(
            model='nomic-embed-text',
            base_url="http://host.docker.internal:11434"
        )

        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name='production_docs',
            connection=settings.database_url,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):

        def route_query(state: AgentState) -> dict:
            query = state['messages'][-1].content

            prompt = f"""Classify this query into one of two categories:
- "retrieval": requires searching documents for specific information
- "conversational": simple chat, greetings, or general knowledge

Query: {query}

Respond with ONLY one word: retrieval or conversational"""

            response = self.primary_llm.invoke([HumanMessage(content=prompt)])
            route = response.content.strip().lower()

            if route not in ['retrieval', 'conversational']:
                route = 'retrieval'  # default to retrieval if unsure

            return {'route': route, 'context': ''}

        def retrieve(state: AgentState) -> dict:
            query = state['messages'][-1].content
            docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return {'context': context}

        def process_message(state: AgentState) -> dict:
            try:
                messages = state['messages']

                if state.get('context'):
                    system = f"""Answer using this context:

{state['context']}

If the context doesn't contain the answer, say so."""
                    messages = [HumanMessage(content=system)] + list(messages)

                response = self.primary_llm.invoke(messages)
                return {
                    'messages': [response],
                    'error': None,
                    'model_used': 'primary'
                }

            except Exception as e:
                return {
                    'error': str(e),
                    'retry_count': state['retry_count'] + 1,
                    'model_used': ''
                }

        def try_fallback(state: AgentState) -> dict:
            try:
                response = self.fallback_llm.invoke(state['messages'])
                return {
                    'messages': [response],
                    'error': None,
                    'model_used': 'fallback'
                }
            except Exception as e:
                return {
                    'error': str(e),
                    'retry_count': state['retry_count'] + 1,
                    'model_used': ''
                }

        def handle_error(state: AgentState) -> dict:
            return {
                'messages': [
                    AIMessage(content='I am sorry, I am having trouble processing your request right now. Please try again later.')
                ],
                'model_used': 'error_handler'
            }

        def route_after_query(state: AgentState) -> str:
            return state.get('route', 'retrieval')

        def route_after_process(state: AgentState) -> str:
            if state.get('error') is None:
                return 'done'
            elif state['retry_count'] < self.max_retries:
                return 'fallback'
            else:
                return 'error'

        def route_after_fallback(state: AgentState) -> str:
            if state.get('error') is None:
                return 'done'
            else:
                return 'error'

        graph = StateGraph(AgentState)

        graph.add_node('route_query', route_query)
        graph.add_node('retrieve', retrieve)
        graph.add_node('process', process_message)
        graph.add_node('fallback', try_fallback)
        graph.add_node('error', handle_error)

        graph.add_edge(START, 'route_query')

        graph.add_conditional_edges(
            'route_query',
            route_after_query,
            {'retrieval': 'retrieve', 'conversational': 'process'}
        )

        graph.add_edge('retrieve', 'process')

        graph.add_conditional_edges(
            'process',
            route_after_process,
            {'done': END, 'fallback': 'fallback', 'error': 'error'}
        )

        graph.add_conditional_edges(
            'fallback',
            route_after_fallback,
            {'done': END, 'error': 'error'}
        )

        graph.add_edge('error', END)

        return graph.compile()

    @traceable
    def invoke(self, message: str) -> dict:
        result = self.graph.invoke({
            'messages': [HumanMessage(content=message)],
            'error': None,
            'retry_count': 0,
            'model_used': '',
            'route': '',
            'context': ''
        })

        return {
            'response': result['messages'][-1].content,
            'model_used': result.get('model_used', 'unknown'),
            'error': result.get('error')
        }
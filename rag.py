"""
Mental Health RAG Chatbot - Function-based Implementation
A specialized RAG implementation for mental health support using Pinecone vector database

Features:
- Function-based architecture instead of class
- User-specific memory storage with MemorySaver
- Specialized mental health counseling prompts
- Safety features and crisis detection
- Multi-step retrieval for comprehensive responses
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# LangGraph imports
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Pinecone imports
from pinecone import Pinecone

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for models and stores
llm = None
embeddings = None
csv_vector_store = None
pdf_vector_store = None
conversational_graph = None
agent_executor = None
memory_saver = None

# Simple cache for storing recent queries and responses
_query_cache = {}
CACHE_TTL = 300  # 5 minutes
CACHE_MAX_SIZE = 100

# Crisis keywords for safety monitoring
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm", "hurt myself",
    "want to die", "better off dead", "no point living", "ending it all"
]

# Medicine-related keywords
MEDICINE_KEYWORDS = [
    "medicine", "medication", "drug", "pill", "tablet", "prescription",
    "dosage", "side effects", "antidepressant", "anxiety medication",
    "sleeping pills", "painkillers", "antipsychotic", "mood stabilizer"
]

def get_cache_key(query: str, user_id: str) -> str:
    """Generate cache key for query"""
    return f"{user_id}:{hash(query.lower().strip())}"

def get_cached_response(query: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get cached response if available and not expired"""
    cache_key = get_cache_key(query, user_id)
    if cache_key in _query_cache:
        cached_data, timestamp = _query_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
        else:
            # Remove expired cache
            del _query_cache[cache_key]
    return None

def cache_response(query: str, user_id: str, response_data: Dict[str, Any]):
    """Cache response data"""
    cache_key = get_cache_key(query, user_id)

    # Remove oldest cache if at max size
    if len(_query_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_query_cache.keys(), key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]

    _query_cache[cache_key] = (response_data, time.time())

def initialize_models(model_name: str = "gpt-4o-mini"):
    """Initialize chat model and embeddings"""
    global llm, embeddings
    
    # Initialize chat model
    llm = init_chat_model(model_name, model_provider="openai")
    
    # Initialize embeddings model - using text-embedding-3-small for faster performance
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("✅ Models initialized successfully")

def setup_pinecone_connections(csv_index_name: str = "csv-mental-health", 
                              pdf_index_name: str = "pdf-mental-health"):
    """Setup connections to Pinecone vector databases"""
    global csv_vector_store, pdf_vector_store
    
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Connect to CSV index (interview and synthetic data)
    try:
        csv_index = pc.Index(csv_index_name)
        csv_vector_store = PineconeVectorStore(
            embedding=embeddings,
            index=csv_index
        )
        print(f"✅ Connected to CSV vector store: {csv_index_name}")
    except Exception as e:
        print(f"⚠️ Could not connect to CSV index: {e}")
        csv_vector_store = None
    
    # Connect to PDF index (research papers and textbooks)
    try:
        pdf_index = pc.Index(pdf_index_name)
        pdf_vector_store = PineconeVectorStore(
            embedding=embeddings,
            index=pdf_index
        )
        print(f"✅ Connected to PDF vector store: {pdf_index_name}")
    except Exception as e:
        print(f"⚠️ Could not connect to PDF index: {e}")
        pdf_vector_store = None

async def parallel_retrieval(query: str) -> List[Document]:
    """Perform parallel retrieval from both CSV and PDF stores"""
    async def retrieve_csv():
        try:
            if csv_vector_store:
                # In a thread pool since Pinecone operations are sync
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, csv_vector_store.similarity_search, query, 2)
            return []
        except Exception as e:
            print(f"CSV retrieval error: {e}")
            return []

    async def retrieve_pdf():
        try:
            if pdf_vector_store:
                # In a thread pool since Pinecone operations are sync
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, pdf_vector_store.similarity_search, query, 2)
            return []
        except Exception as e:
            print(f"PDF retrieval error: {e}")
            return []

    # Run both retrievals in parallel
    csv_docs, pdf_docs = await asyncio.gather(retrieve_csv(), retrieve_pdf())
    return csv_docs + pdf_docs

def create_retrieval_tools():
    """Create optimized retrieval tools for different data sources"""

    @tool(response_format="content_and_artifact")
    async def retrieve_counseling_data(query: str):
        """Retrieve mental health counseling data from interview and synthetic datasets."""
        if not csv_vector_store:
            return "CSV vector store not available", []

        try:
            # Reduced from k=3 to k=2 for faster retrieval
            retrieved_docs = csv_vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                 f"Type: {doc.metadata.get('category', 'Unknown')}\n"
                 f"Data Source: CSV\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error retrieving counseling data: {e}", []

    @tool(response_format="content_and_artifact")
    async def retrieve_research_data(query: str):
        """Retrieve academic research and clinical information from psychology textbooks and papers."""
        if not pdf_vector_store:
            return "PDF vector store not available", []

        try:
            # Reduced from k=3 to k=2 for faster retrieval
            retrieved_docs = pdf_vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                 f"Category: {doc.metadata.get('category', 'Unknown')}\n"
                 f"Data Source: PDF\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error retrieving research data: {e}", []

    # Create an optimized parallel retrieval tool
    @tool(response_format="content_and_artifact")
    async def retrieve_all_data_parallel(query: str):
        """Retrieve data from both CSV and PDF sources in parallel for faster response."""
        try:
            retrieved_docs = await parallel_retrieval(query)
            if not retrieved_docs:
                return "No documents retrieved", []

            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                 f"Category: {doc.metadata.get('category', 'Unknown')}\n"
                 f"Data Source: {doc.metadata.get('data_source', 'Unknown')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error in parallel retrieval: {e}", []

    tools = [retrieve_counseling_data, retrieve_research_data, retrieve_all_data_parallel]
    print("✅ Optimized retrieval tools setup complete")
    return tools

def get_mental_health_system_prompt() -> str:
    """Get the specialized mental health system prompt"""
    return """You are a compassionate mental health support assistant. Provide short, helpful responses.

**RESPONSE STYLE:**
- Keep answers short and direct (2-3 sentences maximum)
- Be supportive but concise
- Focus on practical guidance
- Avoid lengthy explanations

**SAFETY RULES:**
- Never give medical advice or medication recommendations
- Never diagnose conditions
- Always encourage professional help for medical concerns
- For suicide/self-harm: immediate crisis response

**RESPONSE GUIDELINES:**
1. Acknowledge feelings briefly
2. Offer 1-2 practical suggestions
3. Suggest professional help when needed
4. Keep responses under 100 words

Remember: You provide support, not treatment. Always prioritize user safety."""

def detect_crisis(message: str) -> bool:
    """Detect if a message contains crisis indicators"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in CRISIS_KEYWORDS)

def detect_medicine_question(message: str) -> bool:
    """Detect if a message contains medicine/medication related questions"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in MEDICINE_KEYWORDS)

def get_medicine_response() -> str:
    """Get response for medicine-related questions"""
    return "I'm sorry, please consult a qualified healthcare professional for accurate guidance."

def get_crisis_response() -> str:
    """Get immediate crisis response"""
    return "I'm really sorry you're feeling this way. Please reach out to someone you trust, and contact a mental health professional or your nearest emergency support right away."

def setup_conversational_chain(tools):
    """Setup conversational RAG chain with user-specific memory"""
    global conversational_graph, memory_saver
    
    # Create user-specific memory saver
    memory_saver = MemorySaver()
    
    # Create graph builder
    graph_builder = StateGraph(MessagesState)
    
    # Node 1: Safety check and query processing
    def safety_check_and_query(state: MessagesState):
        """Check for crisis indicators and generate tool calls or direct response."""
        last_message = state["messages"][-1]
        
        # Crisis detection
        if isinstance(last_message, HumanMessage) and detect_crisis(last_message.content):
            crisis_response = AIMessage(content=get_crisis_response())
            return {"messages": [crisis_response]}
        
        # Normal processing with tools
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    # Node 2: Tool execution (retrieval)
    tools_node = ToolNode(tools)
    
    # Node 3: Generate empathetic response using retrieved content
    def generate_mental_health_response(state: MessagesState):
        """Generate specialized mental health response using retrieved context."""
        # Get recent tool messages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        
        # Format retrieved content
        if tool_messages:
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            context_prompt = f"""
**RETRIEVED MENTAL HEALTH INFORMATION:**
{docs_content}

**Instructions:** Use this information to inform your response, but prioritize empathy and person-centered care. Integrate relevant insights naturally while maintaining a supportive tone.
"""
        else:
            context_prompt = "**No specific retrieved context available - provide general mental health support based on your training.**"
        
        # Filter conversation messages (exclude tool calls)
        conversation_messages = []
        for message in state["messages"]:
            if message.type in ("human", "system"):
                conversation_messages.append(message)
            elif message.type == "ai":
                # Only include AI messages that don't have tool calls
                try:
                    if not hasattr(message, 'tool_calls') or not message.tool_calls:
                        conversation_messages.append(message)
                except Exception:
                    # If there's any issue checking tool_calls, include the message
                    conversation_messages.append(message)
            # Skip tool messages completely
        
        # Create the prompt with system message
        system_prompt = get_mental_health_system_prompt() + "\n\n" + context_prompt
        prompt = [SystemMessage(system_prompt)] + conversation_messages
        
        # Generate response
        try:
            response = llm.invoke(prompt)
            return {"messages": [response]}
        except Exception as e:
            print(f"Error generating response: {e}")
            # Return a fallback response
            fallback_response = AIMessage(content="I'm here to support you, but I'm experiencing some technical difficulties right now. Please try rephrasing your question or contact support if the issue persists.")
            return {"messages": [fallback_response]}
    
    # Add nodes to graph
    graph_builder.add_node("safety_and_query", safety_check_and_query)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("generate_response", generate_mental_health_response)
    
    # Set entry point and edges
    graph_builder.set_entry_point("safety_and_query")
    graph_builder.add_conditional_edges(
        "safety_and_query",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate_response")
    graph_builder.add_edge("generate_response", END)
    
    # Compile with user-specific memory
    conversational_graph = graph_builder.compile(checkpointer=memory_saver)
    
    print("✅ Conversational RAG chain with user-specific memory setup complete")

def setup_agent(tools):
    """Setup ReAct agent for complex mental health queries"""
    global agent_executor
    
    # Create agent with user-specific memory
    agent_executor = create_react_agent(
        llm, 
        tools, 
        checkpointer=memory_saver
    )
    print("✅ Mental health RAG agent setup complete")

def initialize_rag_system(csv_index_name: str = "csv-mental-health",
                         pdf_index_name: str = "pdf-mental-health",
                         model_name: str = "gpt-4o-mini"):
    """Initialize the complete RAG system"""
    print("🧠 Initializing Mental Health RAG System...")
    
    # Initialize models
    initialize_models(model_name)
    
    # Setup Pinecone connections
    setup_pinecone_connections(csv_index_name, pdf_index_name)
    
    # Create retrieval tools
    tools = create_retrieval_tools()
    
    # Setup conversational chain
    setup_conversational_chain(tools)
    
    # Setup agent
    setup_agent(tools)
    
    print("✅ Mental Health RAG System ready for conversations!")

def get_user_config(user_id: str) -> Dict[str, Any]:
    """Get configuration for user-specific memory thread"""
    return {"configurable": {"thread_id": f"user_{user_id}"}}

def detect_data_source_from_response(response_content: str) -> str:
    """Detect which data source was used based on the response content"""
    if not response_content:
        return "none"
    
    has_csv = "Data Source: CSV" in response_content
    has_pdf = "Data Source: PDF" in response_content
    
    if has_csv and has_pdf:
        return "both"
    elif has_csv:
        return "csv"
    elif has_pdf:
        return "pdf"
    else:
        return "none"

def get_response(message: str, user_id: str, use_agent: bool = False) -> Dict[str, Any]:
    """
    Optimized get response function with caching and performance improvements

    Args:
        message: User's message
        user_id: Unique user identifier for conversation threading
        use_agent: Whether to use agent mode for complex queries

    Returns:
        Dict with response data including data source information
    """
    try:
        # Check cache first (except for agent mode which needs full context)
        if not use_agent:
            cached_response = get_cached_response(message, user_id)
            if cached_response:
                cached_response["timestamp"] = time.time()  # Update timestamp
                cached_response["status_code"] = 200
                return cached_response

        config = get_user_config(user_id)

        response_data = {
            "user_id": user_id,
            "query": message,
            "response": "",
            "mode": "agent" if use_agent else "counselor",
            "crisis_detected": False,
            "data_source": "none",
            "timestamp": time.time(),
            "status_code": 200
        }

        # Check for crisis
        if detect_crisis(message):
            response_data["crisis_detected"] = True
            response_data["response"] = get_crisis_response()
            response_data["data_source"] = "none"  # Crisis response doesn't use data sources
            return response_data

        # Check for medicine questions
        if detect_medicine_question(message):
            response_data["response"] = get_medicine_response()
            response_data["data_source"] = "none"  # Medicine response doesn't use data sources
            return response_data
        
        # Store all messages to analyze data sources
        all_messages = []
        
        # Choose between conversational chain or agent
        try:
            if use_agent:
                # Agent mode
                events = list(agent_executor.stream(
                    {"messages": [{"role": "user", "content": message}]},
                    stream_mode="values",
                    config=config,
                ))
                if events:
                    last_event = events[-1]
                    if "messages" in last_event and last_event["messages"]:
                        response_data["response"] = last_event["messages"][-1].content
                    all_messages = last_event.get("messages", [])
            else:
                # Counselor mode
                steps = list(conversational_graph.stream(
                    {"messages": [{"role": "user", "content": message}]},
                    stream_mode="values",
                    config=config,
                ))
                if steps:
                    last_step = steps[-1]
                    if "messages" in last_step and last_step["messages"]:
                        response_data["response"] = last_step["messages"][-1].content
                    all_messages = last_step.get("messages", [])
        except Exception as stream_error:
            print(f"Error in conversation stream: {stream_error}")
            response_data["response"] = "I'm experiencing some technical difficulties. Please try again or rephrase your question."
            response_data["data_source"] = "none"
            response_data["status_code"] = 500
            return response_data
        
        # Analyze data sources from tool messages
        data_sources_used = set()
        try:
            for msg in all_messages:
                if hasattr(msg, 'type') and msg.type == "tool":
                    if hasattr(msg, 'content') and msg.content:
                        content = str(msg.content)
                        if "Data Source: CSV" in content:
                            data_sources_used.add("csv")
                        if "Data Source: PDF" in content:
                            data_sources_used.add("pdf")
        except Exception as e:
            print(f"Warning: Error analyzing data sources: {e}")
            # Continue without data source info
        
        # Determine data source
        if len(data_sources_used) > 1:
            response_data["data_source"] = "both"
        elif "csv" in data_sources_used:
            response_data["data_source"] = "csv"
        elif "pdf" in data_sources_used:
            response_data["data_source"] = "pdf"
        else:
            response_data["data_source"] = "none"
        
        # Fallback response if no response generated
        if not response_data["response"]:
            response_data["response"] = "I'm here to help, but I'm having trouble processing your message right now. Could you please try rephrasing your question?"
            response_data["data_source"] = "none"

        # Cache the response for future use (only for non-agent mode)
        if not use_agent and response_data.get("status_code") == 200:
            cache_response(message, user_id, response_data)

        return response_data
        
    except Exception as e:
        return {
            "user_id": user_id,
            "query": message,
            "response": f"I apologize, but I encountered an error while processing your message. Please try again. If the problem persists, please contact support.",
            "error": str(e),
            "mode": "error",
            "crisis_detected": False,
            "data_source": "none",
            "timestamp": time.time(),
            "status_code": 500
        }

def chat_interactive(message: str, user_id: str, use_agent: bool = False):
    """Interactive chat interface for console use"""
    config = get_user_config(user_id)
    
    print(f"\n👤 User ({user_id}): {message}")
    print("=" * 60)
    
    # Choose between conversational chain or agent
    if use_agent:
        print("🤖 Agent Mode: Multi-step retrieval")
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config=config,
        ):
            event["messages"][-1].pretty_print()
    else:
        print("🤖 Counselor Mode: Empathetic response")
        for step in conversational_graph.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()

def get_conversation_summary(user_id: str) -> str:
    """Get a summary of the conversation for continuity"""
    return f"Conversation thread: user_{user_id} - Mental health support session"

def clear_conversation(user_id: str):
    """Clear conversation memory for a user"""
    print(f"🧹 Cleared conversation memory for user: {user_id}")
    # The MemorySaver automatically handles user-specific threads

def interactive_mental_health_chat():
    """Interactive mental health chat session"""
    print("🧠 Mental Health Support Chatbot")
    print("=" * 50)
    print("Welcome! I'm here to provide mental health support and information.")
    print("Type 'quit' to exit, 'agent' to use agent mode, 'clear' to clear conversation.")
    print("=" * 50)
    
    # Initialize the system
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("💡 Make sure your Pinecone indexes are created and API keys are set")
        return
    
    # Start conversation
    user_id = f"session_{int(time.time())}"
    use_agent = False
    
    while True:
        try:
            user_input = input(f"\n💬 You ({user_id}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🙏 Thank you for using Mental Health Support. Take care!")
                break
            elif user_input.lower() == 'agent':
                use_agent = not use_agent
                mode = "Agent" if use_agent else "Counselor"
                print(f"\n🔄 Switched to {mode} mode")
                continue
            elif user_input.lower() == 'clear':
                clear_conversation(user_id)
                user_id = f"session_{int(time.time())}"
                continue
            elif not user_input:
                continue
            
            # Process the message
            chat_interactive(user_input, user_id, use_agent)
            
        except KeyboardInterrupt:
            print("\n\n🙏 Thank you for using Mental Health Support. Take care!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")

def demo_mental_health_scenarios():
    """Demonstrate various mental health conversation scenarios"""
    print("🧠 Mental Health RAG Bot - Demo Scenarios")
    print("=" * 60)
    
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Demo scenarios
    scenarios = [
        {
            "title": "Anxiety Support",
            "message": "I've been feeling really anxious lately and I don't know how to cope with it.",
            "use_agent": False,
            "user_id": "demo_user_1"
        },
        {
            "title": "Depression Information",
            "message": "Can you tell me about the symptoms of depression and what treatment options are available?",
            "use_agent": True,
            "user_id": "demo_user_2"
        },
        {
            "title": "Stress Management",
            "message": "I'm overwhelmed with work and personal life. What are some effective stress management techniques?",
            "use_agent": False,
            "user_id": "demo_user_3"
        },
        {
            "title": "Crisis Support",
            "message": "I'm having thoughts of ending my life and I don't know what to do.",
            "use_agent": False,
            "user_id": "demo_user_4"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print("-" * 40)
        
        chat_interactive(
            scenario['message'], 
            scenario['user_id'], 
            scenario['use_agent']
        )
        
        if i < len(scenarios):
            input("\nPress Enter to continue to next scenario...")
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    # Choose demo or interactive mode
    print("🧠 Mental Health RAG Chatbot")
    print("1. Interactive Chat")
    print("2. Demo Scenarios")
    
    choice = input("\nChoose mode (1 or 2): ").strip()
    
    if choice == "2":
        demo_mental_health_scenarios()
    else:
        interactive_mental_health_chat()
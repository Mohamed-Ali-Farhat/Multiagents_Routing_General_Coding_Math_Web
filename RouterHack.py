import os
import base64
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing_extensions import TypedDict
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import pathlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from typing_extensions import Literal, TypedDict
import pickle
import hashlib
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama  # For using Ollama models






# Set Groq API key
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Text-only LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_coder = ChatGroq(
    model="qwen-2.5-coder-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_general = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

vlm = ChatGroq(
    model="llama-3.2-11b-vision-preview",  # Vision model for image analysis
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



math_llm = ChatOllama(
    model="qwen2-math",  
    temperature=0, 
    max_tokens=None, 
    timeout=None,  
    max_retries=2,  
)


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: str = Field(
        description="The next step in the routing process: 'Code', 'Math', 'Web', or 'General'"
    )
    reason: str = Field(
        description="A brief explanation of why this route was chosen"
    )



embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# Configuration variables
DATA_FOLDER = "data"  # Static data folder
EMBEDDING_CACHE_PATH = os.path.join(DATA_FOLDER, "embedding_cache")
FORCE_REBUILD = 0  # Change to 1 to force rebuilding the embeddings

# Create data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_PATH, exist_ok=True)

def get_file_hash(file_path):
    """Generate a hash of the file contents to check for changes"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def prepare_rag_from_file(file_path):
    """Load and prepare the text file for RAG with caching"""
    # Generate file hash to use as cache identifier
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(EMBEDDING_CACHE_PATH, f"{file_hash}.pkl")
    
    # Check if cache exists and FORCE_REBUILD is not enabled
    if os.path.exists(cache_file) and FORCE_REBUILD == 0:
        #print(f"Loading cached vector store from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            return retriever
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
    
    # If no cache or FORCE_REBUILD is enabled, create new embeddings
    #print(f"Building new vector store for {file_path}")
    
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(vectorstore, f)
    #print(f"Saved vector store to cache: {cache_file}")
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

# Specify the path to your text file
text_file_path = os.path.join(DATA_FOLDER, "test.txt")  # Replace with your file name

# Load or create the retriever
retriever = None


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)


# State - Enhanced to handle local images
class State(TypedDict):
    input: str
    has_image: bool
    image_path: Optional[str]
    image_data: Optional[str]
    decision: str
    reason: str
    output: str
    extracted_content: Optional[str]
    query: str
    type: str
    improved_type: str
    context: str 
    













# Helper function to encode local image to base64
def encode_image_to_base64(image_path):
    """Convert a local image file to base64 encoding"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")


# Function to check if file exists and is an image
def is_valid_image_file(file_path):
    """Check if the file exists and has an image extension"""
    if not os.path.exists(file_path):
        return False
        
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    return file_ext in valid_extensions


# Function to process the input and prepare the state
def process_input(text_input, image_path=None) -> State:
    """Process the input and prepare the initial state"""
    state: State = {
        "input": text_input,
        "has_image": False,
        "image_path": None,
        "image_data": None,
        "decision": "",
        "reason": "",
        "output": "",
        "extracted_content": None
    }
    
    # Process image if provided
    if image_path:
        if not is_valid_image_file(image_path):
            state["output"] = f"Error: The image file '{image_path}' does not exist or is not a valid image."
            return state
            
        try:
            state["has_image"] = True
            state["image_path"] = image_path
            state["image_data"] = encode_image_to_base64(image_path)
        except Exception as e:
            state["output"] = f"Error processing image file: {str(e)}"
            return state
    
    return state


# Enhanced router function to consider if input has a local image
def llm_call_router(state: State):
    """Route the input to the appropriate node based on content and presence of image"""
    
    # If there's an image and the input suggests math/equation analysis
    if state["has_image"] and any(term in state["input"].lower() for term in 
                                ["equation", "math", "formula", "solve", "calculate"]):
        return {
            "decision": "Math", 
            "reason": "Input contains an image and suggests mathematical content analysis"
        }
    
    # If there's an image but no specific math terms, check if it's likely to be code-related
    if state["has_image"] and any(term in state["input"].lower() for term in 
                                ["code", "program", "algorithm", "function", "class", "bug"]):
        return {
            "decision": "Code", 
            "reason": "Input contains an image of code and requests programming assistance"
        }
    
    # For regular routing based on text content
    system_prompt = """
    Route the input to the appropriate service based on the content:
    - Code: For programming, development, or coding-related questions
    - Web: For questions about web technologies, searching, or internet-related or even queries with links
    - Math: For mathematical problems or equations (note: this also handles image-based math content)
    - General: For general knowledge questions or queries that don't fit the above categories
    """
    
    decision = router.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["input"]),
        ]
    )
    
    return {"decision": decision.step, "reason": decision.reason}


# Code expert node
def llm_call_code(state: State):
    """Handle code-related requests using the coding-specialized model"""
    
    # If there's an image (e.g., screenshot of code), use vision model first to extract code
    if state["has_image"] and state["image_data"]:
        # First extract code from the image
        extract_message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract any code or programming content from this image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_data']}"}}
            ]
        )
        
        extracted = vlm.invoke([extract_message])
        state["extracted_content"] = extracted.content
        
        # Then analyze with the code-specialized model
        prompt = f"""You are a professional software developer and coding expert.
        
        The user has provided the following request: {state['input']}
        
        I've extracted this code from their image:
        ```
        {state['extracted_content']}
        ```
        
        Please analyze this code and respond to their request.
        """
        
        result = llm_coder.invoke(prompt)
        return {"output": result.content}
    else:
        # Handle text-only code questions
        prompt = f"""You are a professional software developer and coding expert.
        Please answer this request in detail, providing well-structured and documented code:
        
        {state['input']}
        """
        
        result = llm_coder.invoke(prompt)
        return {"output": result.content}


# Math/Image analysis node
def llm_call_math(state: State):
    """Handle math requests, including equation extraction from images"""
    
    # If there's an image, use the vision model to extract content
    if state["has_image"] and state["image_data"]:
        # Format message for the VLM with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Extract and solve the equation from this image. User query: {state['input']}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_data']}"}}
            ]
        )
        
        # Invoke VLM with image
        result = vlm.invoke([message])
        
        # Store both extracted content and the final output
        return {
            "extracted_content": result.content,
            "output": result.content  # In this case, they're the same
        }
    else:
        # Handle text-only math questions
        result = llm.invoke(f"Solve this mathematical problem or equation: {state['input']}")
        return {"output": result.content}


# Web search node
def llm_call_web(state: State):
    """Handle web-related requests"""
    
    result = llm.invoke(f"Acting as a web search expert, I'll address this query: {state['input']}")
    return {"output": result.content}


# General knowledge node
def llm_call_general(state: State):
    """Handle general knowledge requests using the most capable general model"""
    
    # If there's an image, use the vision model
    if state["has_image"] and state["image_data"]:
        message = HumanMessage(
            content=[
                {"type": "text", "text": state['input']},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_data']}"}}
            ]
        )
        
        result = vlm.invoke([message])
        return {"output": result.content}
    else:
        # Text-only query
        result = llm_general.invoke(f"{state['input']}")
        return {"output": result.content}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    """Route to the appropriate node based on the decision"""
    if state["decision"] == "Code":
        return "llm_call_code"
    elif state["decision"] == "Math":
        return "llm_call_math"
    elif state["decision"] == "Web":
        return "initialize_retriever"
    else:  
        return "llm_call_general"



def initialize_retriever(state: State):
    """Initialize the retriever if not already done"""
    global retriever
    if retriever is None:
        retriever = prepare_rag_from_file(text_file_path)
    return {}  # No state change





def retrieve_context(state: State):
    """Retrieve relevant information from the text file"""
    global retriever
    
    # Initialize retriever if not done yet
    if retriever is None:
        retriever = prepare_rag_from_file(text_file_path)
    
    # Use the query to retrieve relevant context
    docs = retriever.get_relevant_documents(state["query"])
    
    # Extract and combine the content from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    #print(f"Retrieved context: {context[:200]}...")  # Print first 200 chars of context
    
    return {"context": context}




def generate_type_general(state: State):
    """RAG-enhanced function to generate a response for general queries using retrieved context"""
    # Use the retrieved context to enhance the response
    prompt = f"""
    Based on the following information from our knowledge base:
    
    {state['context']}
    
    Please provide an response for the query: {state['query']}
    
    
    
    """
    
    msg = llm.invoke(prompt)
    print(msg.content)
    return {"improved_type": msg.content}











# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("llm_call_code", llm_call_code)
router_builder.add_node("llm_call_math", llm_call_math)
router_builder.add_node("llm_call_web", llm_call_web)
router_builder.add_node("llm_call_general", llm_call_general)

router_builder.add_node("initialize_retriever", initialize_retriever)
router_builder.add_node("retrieve_context", retrieve_context)
router_builder.add_node("generate_type_general", generate_type_general)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_code": "llm_call_code",
        "llm_call_math": "llm_call_math",
        "initialize_retriever": "initialize_retriever",
        "llm_call_general": "llm_call_general",
    },
)
router_builder.add_edge("llm_call_code", END)
router_builder.add_edge("llm_call_math", END)
router_builder.add_edge("llm_call_general", END)


router_builder.add_edge("initialize_retriever", "retrieve_context")
router_builder.add_edge("retrieve_context", "generate_type_general")  
router_builder.add_edge("generate_type_general", END)  



# Compile workflow
router_workflow = router_builder.compile()


# Main function to run the workflow with local image
def process_query(text_input, image_path=None):
    """Process a user query with optional local image input"""
    
    # Initialize the state with the input and image path
    initial_state = process_input(text_input, image_path)
    
    # If there was an error processing the image
    if initial_state["output"]:
        return {"error": initial_state["output"]}
    
    # Run the workflow
    result = router_workflow.invoke(initial_state)
    
    return {
        "query": text_input,
        "has_image": initial_state["has_image"],
        "image_path": initial_state.get("image_path"),
        "routed_to": result["decision"],
        "routing_reason": result.get("reason", ""),
        "response": result["output"],
        "extracted_content": result.get("extracted_content")
    }


# Example usage:


img=True

if img :
    text_result = process_query(" what is the theory behind the equation in this image ? ", image_path="2.jpg")
else :
    text_result = process_query(" what is the documentation here in this link https://docs.unity3d.com  ? ")



print(text_result.get("response", "No response found."))


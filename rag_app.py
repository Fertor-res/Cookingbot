

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

# configure the page 
st.set_page_config(
    page_title="Find your recipes!",
    page_icon="🍳",
    layout="centered",
)
# Apply colors via CCS

st.markdown("""
<style>
:root {
    --primary-color: #f49a0aff;
    --background-color: #ffffffaa;
    --text-color: #056ce9ff;
}
</style>
""", unsafe_allow_html=True)


# Changing the chat background for an image:
from pathlib import Path
import base64

# Function to change the entire app background
def set_app_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def set_chat_background(image_file):
    import base64
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stChatMessage {{
        /* Layer 1: gradient overlay, Layer 2: your image */
        background-image:
            linear-gradient(to bottom, rgba(255,255,255,0.9), rgba(255,255,255,0.5)),
            url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Full app background
full_bg_path = Path(__file__).parent / "kitchen_wall.jpg"
set_app_background(str(full_bg_path))

# Set the chat background
image_path = Path(__file__).parent / "kitchen_background.jpg"
set_chat_background(str(image_path))





### INITIALIZING AND CACHING CHATBOT COMPONENTS ###

# Function for initializing the LLM
@st.cache_resource #the result will be cached so it only has to rerun when temp changes
def init_llm(temp=0.01):
    # LLM
    return Groq(
    model="llama-3.3-70b-versatile",
    max_new_tokens=768,
    temperature=temp,
    top_p=0.95,
    repetition_penalty=1.03,
    token=st.secrets["GROQ_API_KEY"]
    )

# Function for initializing the retriever
@st.cache_resource #the result will be cached so it only has to rerun when num_chunks changes
def init_rag(num_chunks=2):
    # RAG
    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        #cache_folder="./embedding_model/",
    )
    storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    return vector_index.as_retriever(similarity_top_k=num_chunks)


# Function for initializing the chatbot memory
@st.cache_resource #the result will be cached so it only has to run once
def init_memory():
    return ChatMemoryBuffer.from_defaults()


# Function for initializing the bot with the specific settings
@st.cache_resource
def init_bot(prefix_messages, temp=0.01, num_chunks=2):
    # Initialize components
    llm = init_llm(temp)
    retriever = init_rag(num_chunks)
    memory = init_memory()

    # Build ChatMessage list safely
    safe_prefix_messages = []
    for system_prompt_selection in prefix_messages:
        if system_prompt_selection in prompt_options:
            # Use the predefined prompt text
            content = prompt_options[system_prompt_selection]
        else:
            # Use the string as-is (dynamic prompt)
            content = system_prompt_selection

        safe_prefix_messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=content
            )
        )

    # Return initialized bot
    return ContextChatEngine(
        llm=llm,
        retriever=retriever,
        memory=memory,
        prefix_messages=safe_prefix_messages
    )


    ##### STREAMLIT #####

st.title("Your Virtual Chef!")


### PROMPT CUSTOMIZATION ###


# Create two side by side columns
col1, col2 = st.columns(2)

# Cooking confidence (Beginner/Expert) -- Using radio

with col1:
    st.radio('How confident are you with cooking? ', ['Beginner', 'Expert'], key= "cooking_mode")
    

# Recipe type (Starter / Main Dish / Dessert) — using segmented control

with col2:
    st.segmented_control(
        "Recipe type:",
        ["Starter", "Main Dish", "Dessert"],
        key="recipe_type",
        default="Main Dish"  # 👈 ensures a value immediately
    )

selected_mode = st.session_state.get("cooking_mode", "Beginner")
selected_type = st.session_state.get("recipe_type", "Main Dish")

st.session_state['system_prompts'] = [
    "basic_context",
    selected_mode,
    f"You are preparing a {selected_type.lower()} recipe. Adjust your suggestions accordingly."
]

# Setting up session state to store current system prompt setting
if 'system_prompts' not in st.session_state:
    st.session_state['system_prompts'] = ['basic_context', selected_mode, f"You are preparing a {selected_type.lower()} recipe. Adjust your suggestions accordingly."
] #making it a list allow it to have multiple at once


# Setting up system prompt options:
prompt_options = {
    'basic_context': (
        'You are a chatbot with two modes: Beginner and Expert. '
        f"You are preparing a {selected_type.lower()} recipe. Adjust your suggestions accordingly."
        'You are a helpful chatbot having a conversation with a human. '
        'Give priority to the recipes feed to you in the .pdf files provided to you'
        "Everytime you are queried to do something outside the topic cooking, answer with 'Sorry, I can only cook'. Do not answer outside the world of cooking."
        "At the bottom of the recipe, please provide the source from where you took the information"
        ),
    'Beginner': (
        'YOU ARE NOW IN BEGINNER MODE, change your behavior if needed. '
        'You are a helpful chatbot having a conversation with a human. '
        'Look for easy to cook recipes. Ask always the number of people the user will cook for. Adjust your answer to that number of people. Present the ingredients first in bullet points, after that, the cooking instructions as detailed as possible and include preparation times'
        ),
    'Expert': (
        'YOU ARE NOW IN EXPERT MODE, change your behavior if needed. '
        'Search for complex recipes. Ask always the number of people the user will cook for. Adjust your answer to that number of people. Present the ingredients first in bullet points, after that, the cooking instructions with a low level of detail unless the user specify otherwise and include preparation times '
        )
}


### CHAT ###

# Initializing chatbot
# If the parameters change, this reruns, otherwise it uses what is in the cache already
rag_bot = init_bot(
    prefix_messages=st.session_state['system_prompts'],
    temp=0.5,
    num_chunks=2
)

# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)


# React to user input
if prompt := st.chat_input('Reset the chat by typing "Goodbye"'):

    # If user types "goodbye", reset the memory and run the app from the top again
    if prompt.lower() == 'goodbye':
        rag_bot.reset() # reset the bot memory
        st.rerun() # reruns the app so that the bot is reinitialized and the chat is cleared
    
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Be patient, a good meal requires always time!..."):
        # send question to bot to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from bot's response
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
# Use streamlit run rag_app.py in Terminal to run this Python code

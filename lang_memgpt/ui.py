import streamlit as st
import speech_recognition as sr
from audiorecorder import audiorecorder
import tempfile
import os
import asyncio
import uuid

# If these imports differ in your code, adjust accordingly
from lang_memgpt.graph import memgraph
from lang_memgpt._schemas import State, GraphConfig

# ------------------------------------------------------------------------------
# SESSION / STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_session():
    """Initialize the session keys for state and config as dictionaries."""
    if "chat_state" not in st.session_state:
        # If you actually need the 'State' object, store it here.
        st.session_state.chat_state = State(messages=[])
    if "config" not in st.session_state:
        # Store config as a dictionary to avoid 'dict object has no attribute x'
        st.session_state.config = {
            "configurable": {
                "model": "gpt-4",
                "user_id": "default_user",
                "thread_id": str(uuid.uuid4())
            }
        }

# ------------------------------------------------------------------------------
# AUDIO PROCESSING
# ------------------------------------------------------------------------------
def process_audio():
    """Process audio from 'audiorecorder' and convert to text via SpeechRecognition."""
    r = sr.Recognizer()
    audio_bytes = st.session_state.audio_recorder

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_bytes)
            temp_filename = fp.name

        with sr.AudioFile(temp_filename) as source:
            audio = r.record(source)
            try:
                text = r.recognize_google(audio)
                return text
            except Exception as e:
                st.error(f"Error recognizing speech: {str(e)}")
                return None
            finally:
                os.unlink(temp_filename)
    return None

# ------------------------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------------------------
def process_uploaded_file(uploaded_file):
    """Handle file upload and pass it to memgraph.ainvoke for ingestion."""
    if uploaded_file:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            message = {
                "messages": [{"content": f"Please process this file: {file_path}", "type": "user"}],
                "configurable": st.session_state.config["configurable"]
            }
            asyncio.run(memgraph.ainvoke(message))
            st.success("File processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            os.unlink(file_path)

# ------------------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------------------
def set_custom_style():
    """
    Custom CSS ensuring:
    1. Chat container has a fixed height and is scrollable.
    2. Input area is pinned at the bottom.
    3. Title remains visible at the top.
    """
    st.markdown(
        """
        <style>
        /* Make the main container auto, so the page can grow if needed */
        body, .block-container {
            margin: 0;
            padding: 0;
            height: auto !important;
            overflow: visible !important;
        }
        
        /* Title margin fix */
        .css-10trblm.e16nr0p33 {
            margin-top: 1rem;
        }
        
        /* Chat container: fixed height, scroll inside */
        .chat-container {
            max-height: 60vh;
            overflow-y: auto;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 4px;
            margin-bottom: 1rem;
        }

        /* Message bubbles */
        .user-message {
            background-color: #e2e2e2;
            color: #111;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .assistant-message {
            background-color: #464646;
            color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }

        /* Pin input area at bottom */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #ffffff;
            padding: 1rem;
            border-top: 1px solid #ddd;
            z-index: 1000;
        }

        /* Slight spacing improvement for text input & audio button */
        .stTextInput input {
            min-height: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# MAIN STREAMLIT APP
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Alfred - The Future AI Agent",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/alfred',
            'Report a bug': "https://github.com/yourusername/alfred/issues",
            'About': "# Alfred AI Assistant\nA futuristic AI agent with memory."
        }
    )

    set_custom_style()
    initialize_session()
    
    st.title("Alfred — Your Future AI Agent")

    # Sidebar: File upload
    with st.sidebar:
        st.header("Manage Your Data")
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'csv', 'xlsx', 'doc', 'docx'])
        if uploaded_file:
            process_uploaded_file(uploaded_file)

    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            message_class = "user-message" if role == "user" else "assistant-message"
            content = message.get("content", "")
            st.markdown(f'<div class="{message_class}">{content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fixed input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1], gap="small")
    with col1:
        user_input = st.text_input("Your message to Alfred", key="user_input")
    with col2:
        # Audio recorder
        st.write("")  # Slight spacer
        audio_recorder_data = audiorecorder("🎤", "Recording...")
    st.markdown('</div>', unsafe_allow_html=True)

    # Store audio recorder data in session
    st.session_state.audio_recorder = audio_recorder_data
    
    # Process recorded audio
    if audio_recorder_data and len(audio_recorder_data) > 0:
        recognized_text = process_audio()
        if recognized_text:
            user_input = recognized_text
            st.session_state.user_input = recognized_text

    # Process user text input
    if user_input:
        # Append user's message
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Build request for AI
            message = {
                "messages": st.session_state.messages,
                "configurable": st.session_state.config["configurable"]
            }
            new_state = asyncio.run(memgraph.ainvoke(message))

            # If AI responds with a dict containing "messages"
            if isinstance(new_state, dict) and "messages" in new_state:
                ai_message = new_state["messages"]
                # If it's a single message or content object
                if hasattr(ai_message, 'content'):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_message.content
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(ai_message)
                    })

            # Clear the user input box
            st.session_state.user_input = ""

        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

        # Rerun to update chat
        st.rerun()

if __name__ == "__main__":
    main()

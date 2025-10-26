import streamlit as st
import os
import sys
import time
import tempfile
import warnings
import speech_recognition as sr
from gtts import gTTS
from google import genai
from dotenv import load_dotenv
import re
from io import BytesIO
import numpy as np
import sounddevice as sd
import wave
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Suppress pygame output
import contextlib
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

# Page configuration
st.set_page_config(
    page_title="Friday Voice Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'recorded_audio_path' not in st.session_state:
    st.session_state.recorded_audio_path = None

# Configure Gemini API
@st.cache_resource
def initialize_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return None, "API key not found"
    try:
        client = genai.Client(api_key=api_key)
        MODEL_NAME = 'gemini-2.0-flash-exp'
        # Test connection
        test_response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Say 'OK'"
        )
        return client, MODEL_NAME
    except Exception as e:
        return None, str(e)

client, model_or_error = initialize_gemini()
if client:
    st.session_state.api_connected = True
    MODEL_NAME = model_or_error
else:
    st.session_state.api_connected = False
    st.session_state.error_message = model_or_error

# Audio Recording Class
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self.recording = False
        self.stream = None
    
    def callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(status)
        self.frames.append(indata.copy())
    
    def start(self):
        """Start recording"""
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback
        )
        self.stream.start()
        return True
    
    def stop(self):
        """Stop recording and return audio data"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.recording = False
        
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            return audio_data
        return None
    
    def save_recording(self, audio_data, filename):
        """Save recorded audio to WAV file"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        filepath = os.path.join("data", f"{filename}.wav")
        
        # Convert float32 to int16
        audio_int16 = np.int16(audio_data * 32767)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return filepath

# Initialize recorder
if 'recorder' not in st.session_state:
    st.session_state.recorder = AudioRecorder()

# Helper Functions
def clean_text_for_speech(text):
    """Remove Markdown formatting"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_to_speech(text, lang='en'):
    """Convert text to speech and return audio bytes"""
    try:
        clean_text = clean_text_for_speech(text)
        speech = gTTS(text=clean_text, lang=lang, slow=False, tld="com.au")
        
        # Save to BytesIO instead of file
        audio_buffer = BytesIO()
        speech.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None

def transcribe_audio_file(audio_file_path):
    """Transcribe audio file to text"""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        return text, None
    except Exception as e:
        return None, str(e)

def transcribe_uploaded_audio(audio_file):
    """Transcribe uploaded audio file to text"""
    r = sr.Recognizer()
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return text, None
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, str(e)

def get_ai_response(question, short=True):
    """Get response from Gemini"""
    try:
        if short:
            full_prompt = f"""You are Friday, a voice assistant. Give VERY short, crisp answers (maximum 10-15 words).
Be direct and concise. No extra explanations unless asked.

Question: {question}

Answer briefly:"""
        else:
            full_prompt = question
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )
        
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_text(content):
    """Summarize provided text"""
    prompt = f"Summarize this in 2-3 clear sentences:\n\n{content}"
    return get_ai_response(prompt, short=False)

# Main UI
st.title("🎤 Friday Voice Assistant")
st.markdown("### Your AI-powered voice assistant with Gemini")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Status
    if st.session_state.api_connected:
        st.success("✅ Connected to Gemini API")
        st.info(f"📡 Model: {MODEL_NAME}")
    else:
        st.error("❌ API Connection Failed")
        st.error(f"Error: {st.session_state.error_message}")
        st.info("💡 Make sure your .env file contains GEMINI_API_KEY")
    
    st.markdown("---")
    
    # Voice Settings
    st.subheader("🔊 Voice Settings")
    enable_audio = st.checkbox("Enable Audio Responses", value=True)
    response_mode = st.radio(
        "Response Length",
        ["Short (10-15 words)", "Detailed"],
        index=0
    )
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Chat Tab**: Ask questions via text or voice
    2. **Summarize Notes**: Upload text files to summarize
    3. **Transcribe Audio**: Convert audio files to text
    """)
    
    st.markdown("---")
    st.markdown("### 🎙️ Recording Tips")
    st.markdown("""
    - Click **Start Recording** to begin
    - Speak clearly into your microphone
    - Click **Stop Recording** when done
    - Click **Save & Transcribe** to process
    """)

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summarize Notes", "🎵 Transcribe Audio"])

# TAB 1: Chat Interface
with tab1:
    st.subheader("Ask Friday Anything")
    
    # Recording Controls
    st.markdown("#### 🎙️ Voice Recording")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        if st.button("🔴 Start Recording", disabled=st.session_state.recording, use_container_width=True):
            try:
                st.session_state.recorder.start()
                st.session_state.recording = True
                st.success("🎙️ Recording started...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Recording failed: {str(e)}")
    
    with col_rec2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording, use_container_width=True):
            try:
                audio_data = st.session_state.recorder.stop()
                st.session_state.audio_frames = audio_data
                st.session_state.recording = False
                st.success("✅ Recording stopped")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Stop failed: {str(e)}")
    
    with col_rec3:
        if st.button("💾 Save & Transcribe", 
                    disabled=st.session_state.recording or st.session_state.audio_frames is None,
                    use_container_width=True):
            try:
                if st.session_state.audio_frames is not None:
                    # Save recording
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}"
                    audio_path = st.session_state.recorder.save_recording(
                        st.session_state.audio_frames,
                        filename
                    )
                    st.session_state.recorded_audio_path = audio_path
                    st.success(f"✅ Saved: {os.path.basename(audio_path)}")
                    
                    # Transcribe
                    with st.spinner("🎧 Transcribing..."):
                        text, error = transcribe_audio_file(audio_path)
                        
                        if text:
                            st.success(f"📝 You said: **{text}**")
                            
                            # Process with AI if connected
                            if st.session_state.api_connected:
                                short_mode = response_mode == "Short (10-15 words)"
                                with st.spinner("🤖 Friday is thinking..."):
                                    response = get_ai_response(text, short=short_mode)
                                    
                                    # Add to chat history
                                    st.session_state.chat_history.append({
                                        "role": "user",
                                        "content": text
                                    })
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": response
                                    })
                                    
                                    # Reset audio frames
                                    st.session_state.audio_frames = None
                                    st.rerun()
                        else:
                            st.error(f"❌ Transcription failed: {error}")
                else:
                    st.warning("⚠️ No recording available")
            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")
    
    # Show recording status
    if st.session_state.recording:
        st.warning("🔴 **Recording in progress...** Click 'Stop Recording' when done.")
    
    # Show last recorded audio
    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
        with st.expander("🎵 Last Recording"):
            st.audio(st.session_state.recorded_audio_path, format="audio/wav")
    
    st.markdown("---")
    
    # Alternative: Upload Audio
    st.markdown("#### 📁 Or Upload Audio File")
    uploaded_audio = st.file_uploader("Upload audio file (.wav)", type=["wav"], key="chat_audio_upload")
    
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("🎧 Transcribe Uploaded Audio", key="transcribe_chat_upload"):
            with st.spinner("🎧 Transcribing..."):
                text, error = transcribe_uploaded_audio(uploaded_audio)
                
                if text:
                    st.success(f"📝 Transcribed: **{text}**")
                    
                    # Process with AI
                    if st.session_state.api_connected:
                        short_mode = response_mode == "Short (10-15 words)"
                        with st.spinner("🤖 Friday is thinking..."):
                            response = get_ai_response(text, short=short_mode)
                            
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": text
                            })
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            st.rerun()
                else:
                    st.error(f"❌ Transcription error: {error}")
    
    st.markdown("---")
    
    # Text Input Section
    st.markdown("#### ✍️ Text Input")
    user_question = st.text_input(
        "Or type your question here:",
        placeholder="Ask me anything...",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Send 📤", use_container_width=True)
    
    if submit_button and user_question:
        if st.session_state.api_connected:
            short_mode = response_mode == "Short (10-15 words)"
            
            with st.spinner("🤖 Friday is thinking..."):
                response = get_ai_response(user_question, short=short_mode)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
        else:
            st.error("Please configure API key first!")
    
    # Display chat history
    st.markdown("---")
    st.subheader("💬 Conversation History")
    
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
                    
                    # Audio playback option for latest message
                    if enable_audio and i == len(st.session_state.chat_history) - 1:
                        with st.spinner("🔊 Generating audio..."):
                            audio_bytes = text_to_speech(message['content'])
                            if audio_bytes:
                                st.audio(audio_bytes, format='audio/mp3')
    else:
        st.info("👋 Start a conversation with Friday! Use voice recording, upload audio, or type your question.")

# TAB 2: Summarize Notes
with tab2:
    st.subheader("📝 Summarize Text Files")
    st.markdown("Upload a text file or paste text directly to get a concise summary.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'text'],
            help="Upload .txt, .md, or other text files",
            key="notes_file_upload"
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                st.success(f"✅ Loaded {len(content)} characters from {uploaded_file.name}")
                
                # Show preview
                with st.expander("📄 Preview file content"):
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))
                
                if st.button("📊 Summarize File", use_container_width=True, key="summarize_file"):
                    if st.session_state.api_connected:
                        with st.spinner("🤖 Generating summary..."):
                            summary = summarize_text(content)
                            
                            st.markdown("#### 📋 Summary:")
                            st.success(summary)
                            
                            # Audio option
                            if enable_audio:
                                with st.spinner("🔊 Generating audio..."):
                                    audio_bytes = text_to_speech(summary)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format='audio/mp3')
                    else:
                        st.error("API not connected!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("#### ✍️ Paste Text")
        text_input = st.text_area(
            "Or paste your text here:",
            height=250,
            placeholder="Paste your notes, articles, or any text you want summarized...",
            key="notes_text_input"
        )
        
        if text_input:
            st.info(f"📊 {len(text_input)} characters, {len(text_input.split())} words")
            
            if st.button("📊 Summarize Text", use_container_width=True, key="summarize_text"):
                if st.session_state.api_connected:
                    with st.spinner("🤖 Generating summary..."):
                        summary = summarize_text(text_input)
                        
                        st.markdown("#### 📋 Summary:")
                        st.success(summary)
                        
                        # Audio option
                        if enable_audio:
                            with st.spinner("🔊 Generating audio..."):
                                audio_bytes = text_to_speech(summary)
                                if audio_bytes:
                                    st.audio(audio_bytes, format='audio/mp3')
                else:
                    st.error("API not connected!")

# TAB 3: Transcribe Audio
with tab3:
    st.subheader("🎵 Audio Transcription")
    st.markdown("Record audio or upload a file to convert speech to text.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎙️ Record Audio")
        
        # Recording controls for transcription tab
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            if st.button("🔴 Start", disabled=st.session_state.recording, use_container_width=True, key="transcribe_start"):
                try:
                    st.session_state.recorder.start()
                    st.session_state.recording = True
                    st.success("🎙️ Recording...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col_t2:
            if st.button("⏹️ Stop", disabled=not st.session_state.recording, use_container_width=True, key="transcribe_stop"):
                try:
                    audio_data = st.session_state.recorder.stop()
                    st.session_state.audio_frames = audio_data
                    st.session_state.recording = False
                    st.success("✅ Stopped")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col_t3:
            if st.button("💾 Save", 
                        disabled=st.session_state.recording or st.session_state.audio_frames is None,
                        use_container_width=True,
                        key="transcribe_save"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"transcribe_{timestamp}"
                    audio_path = st.session_state.recorder.save_recording(
                        st.session_state.audio_frames,
                        filename
                    )
                    st.session_state.recorded_audio_path = audio_path
                    st.success(f"✅ Saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Show recording status
        if st.session_state.recording:
            st.warning("🔴 Recording...")
        
        # Show and transcribe recorded audio
        if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
            st.audio(st.session_state.recorded_audio_path, format="audio/wav")
            
            if st.button("📝 Transcribe Recording", use_container_width=True, key="do_transcribe_recording"):
                with st.spinner("🎧 Transcribing..."):
                    text, error = transcribe_audio_file(st.session_state.recorded_audio_path)
                    
                    if text:
                        st.success("✅ Transcription complete!")
                        st.markdown("#### 📄 Transcription:")
                        st.info(text)
                        
                        # Copy to clipboard
                        st.code(text, language=None)
                        
                        # Option to summarize
                        if st.button("📊 Summarize This", key="sum_recording"):
                            if st.session_state.api_connected:
                                with st.spinner("🤖 Generating summary..."):
                                    summary = summarize_text(text)
                                    st.markdown("#### 📋 Summary:")
                                    st.success(summary)
                                    
                                    if enable_audio:
                                        audio_bytes = text_to_speech(summary)
                                        if audio_bytes:
                                            st.audio(audio_bytes, format='audio/mp3')
                            else:
                                st.error("API not connected!")
                    else:
                        st.error(f"❌ Transcription failed: {error}")
    
    with col2:
        st.markdown("#### 📁 Upload Audio File")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Supported formats: WAV, MP3, OGG, FLAC",
            key="transcribe_file_upload"
        )
        
        if audio_file:
            st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
            
            if st.button("📝 Transcribe File", use_container_width=True, key="transcribe_uploaded_file"):
                with st.spinner("🎧 Transcribing audio..."):
                    text, error = transcribe_uploaded_audio(audio_file)
                    
                    if text:
                        st.success("✅ Transcription complete!")
                        st.markdown("#### 📄 Transcription:")
                        st.info(text)
                        
                        # Copy to clipboard
                        st.code(text, language=None)
                        
                        # Option to summarize
                        if st.button("📊 Summarize Transcription", key="sum_file"):
                            if st.session_state.api_connected:
                                with st.spinner("🤖 Generating summary..."):
                                    summary = summarize_text(text)
                                    st.markdown("#### 📋 Summary:")
                                    st.success(summary)
                                    
                                    if enable_audio:
                                        audio_bytes = text_to_speech(summary)
                                        if audio_bytes:
                                            st.audio(audio_bytes, format='audio/mp3')
                            else:
                                st.error("API not connected!")
                    else:
                        st.error(f"❌ Transcription failed: {error}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🤖 Friday Voice Assistant | Powered by Google Gemini & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
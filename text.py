import os
import sys
import time
import warnings
import speech_recognition as sr
from gtts import gTTS
from google import genai
from dotenv import load_dotenv
import pygame
import re

# Suppress ALL warnings and ALSA errors
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore")

# Redirect ALSA errors to /dev/null (Linux only)
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# Initialize pygame for audio playback
pygame.mixer.init()

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-2.0-flash-exp'

# Test connection
try:
    test_response = client.models.generate_content(
        model=MODEL_NAME,
        contents="Say 'OK'"
    )
    print(f"✅ Connected to Gemini API")
    print(f"✅ Using model: {MODEL_NAME}\n")
except Exception as e:
    print(f"❌ API Connection Failed: {e}")
    exit(1)

lang = 'en'

print("🎤 Friday Voice Assistant - Continuous Mode!")
print("=" * 60)
print("How to use:")
print("  1. Say 'Friday' ONCE to activate")
print("  2. Keep asking questions - no need to say Friday again!")
print("  3. Say 'stop' or 'exit' to quit anytime")
print()
print("Special commands:")
print("  • 'summarize' - I'll ask for the file path")
print("  • 'transcribe audio' - Convert audio file to text")
print()
print("📁 Default notes folder: ~/Desktop/")
print("   Just put your notes.txt on Desktop and say 'summarize'!")
print("=" * 60)
print("\n👂 Listening for 'Friday' to activate...\n")

def listen_for_wake_word():
    """Listen for wake word to activate"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        r.energy_threshold = 4000
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            text = r.recognize_google(audio).lower()
            
            if "friday" in text:
                return True
            elif "stop" in text or "exit" in text:
                return "stop"
            
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            pass
        except Exception:
            pass
    
    return False

def get_user_question():
    """Listen for question"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎧 I'm listening...")
        r.adjust_for_ambient_noise(source, duration=0.3)
        r.energy_threshold = 4000
        
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            question = r.recognize_google(audio)
            print(f"📝 You said: {question}")
            return question
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("❌ Couldn't understand audio")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def read_text_file(filepath):
    """Read text from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def transcribe_audio_file(filepath):
    """Transcribe audio file to text"""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(filepath) as source:
            print("🎵 Reading audio file...")
            audio = r.record(source)
            print("🔄 Transcribing...")
            text = r.recognize_google(audio)
            return text
    except Exception as e:
        return f"Error transcribing audio: {e}"

def summarize_notes():
    """Handle note summarization with multiple options"""
    print("\n📄 Note Summarization")
    print("=" * 50)
    
    # Check common locations first
    common_files = [
        os.path.expanduser("~/Desktop/notes.txt"),
        os.path.expanduser("~/Desktop/note.txt"),
        os.path.expanduser("~/Documents/notes.txt"),
        "./notes.txt",
        "./note.txt"
    ]
    
    found_file = None
    for filepath in common_files:
        if os.path.exists(filepath):
            found_file = filepath
            print(f"✅ Found: {filepath}")
            response = input("Use this file? (yes/no): ").lower()
            if response == "yes" or response == "y":
                break
            else:
                found_file = None
    
    # If no file found, ask for path
    if not found_file:
        print("\nOptions:")
        print("  1. Enter file path")
        print("  2. Paste text directly")
        print("  3. Drag and drop file here")
        
        user_input = input("\nYour choice: ").strip()
        
        # Check if it's a file path
        if os.path.exists(user_input):
            found_file = user_input
        else:
            # It's direct text
            print(f"✅ Using direct text ({len(user_input)} characters)")
            content = user_input
    
    # Read the file if we have one
    if found_file:
        content = read_text_file(found_file)
        print(f"✅ Loaded {len(content)} characters from file")
    
    if content and len(content) > 10:
        print("🤖 Generating summary...\n")
        prompt = f"Summarize this in 2-3 clear sentences:\n\n{content}"
        response = get_ai_response(prompt, short=False)
        return response
    else:
        return "No content to summarize."

def transcribe_audio():
    """Handle audio transcription"""
    print("\n🎵 Audio Transcription")
    filepath = input("Audio file path (.wav): ").strip()
    
    if not os.path.exists(filepath):
        return "File not found."
    
    text = transcribe_audio_file(filepath)
    print(f"📝 Transcription: {text}")
    
    # Ask if user wants summary
    print("\nWant a summary? (yes/no)")
    if input().lower() == "yes":
        prompt = f"Summarize this in 2-3 sentences:\n\n{text}"
        return get_ai_response(prompt, short=False)
    
    return text

def get_ai_response(question, short=True):
    """Get response from Gemini"""
    try:
        print(f"🤖 Processing...")
        
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
        
        text = response.text
        print(f"💬 Friday: {text}\n")
        return text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        return "Sorry, couldn't process that."

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

def speak_response(text):
    """Convert to speech and play"""
    try:
        clean_text = clean_text_for_speech(text)
        speech = gTTS(text=clean_text, lang=lang, slow=False, tld="com.au")
        speech.save("response.mp3")
        
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        time.sleep(0.1)
        
        if os.path.exists("response.mp3"):
            os.remove("response.mp3")
            
    except Exception as e:
        print(f"❌ Speech error: {e}")

# Main loop
try:
    # First, wait for wake word
    print("🔊 Say 'Friday' to activate...")
    
    while True:
        wake_status = listen_for_wake_word()
        
        if wake_status == "stop":
            print("\n👋 Goodbye!")
            break
        
        if wake_status:
            print("\n✅ Friday activated! I'm ready for your questions.")
            print("💡 No need to say 'Friday' again - just keep asking!\n")
            
            # Now stay active and keep listening
            while True:
                question = get_user_question()
                
                if question:
                    question_lower = question.lower()
                    
                    # Check if user wants to stop
                    if "stop" in question_lower or "exit" in question_lower or "quit" in question_lower:
                        print("\n👋 Goodbye!")
                        pygame.mixer.quit()
                        exit(0)
                    
                    # Check for special commands (support both spellings)
                    elif "summarize" in question_lower or "summarise" in question_lower or "summary" in question_lower:
                        response = summarize_notes()
                        if response:
                            speak_response(response)
                    
                    elif "transcribe" in question_lower and "audio" in question_lower:
                        response = transcribe_audio()
                        if response:
                            speak_response(response)
                    
                    else:
                        # Normal question
                        response = get_ai_response(question)
                        speak_response(response)
                
                # Continue listening for next question
                print("🎧 Ready for next question...\n")
            
except KeyboardInterrupt:
    print("\n\n👋 Interrupted by user. Goodbye!")
finally:
    pygame.mixer.quit()
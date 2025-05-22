import streamlit as st
import os
import requests # For Ollama
import json
import datetime
import random
import sqlite3
import google.generativeai as genai
import re # For parsing AI score
import pandas as pd # For dashboard table and charts

# --- CONFIGURATION ---
# Ollama default settings (can be overridden by user selection)
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:latest" # User can select others if available

DB_NAME = "learning_bot_data.db"
DATA_DIR = "data"
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")
BOOKS_DIR = os.path.join(DATA_DIR, "books")

# Create data directories if they don't exist
os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(BOOKS_DIR, exist_ok=True)


# --- DATABASE SETUP (SQLite) ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        summary TEXT,
        filename TEXT UNIQUE, -- Name of the .txt file in data/books/
        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_description TEXT NOT NULL,
        task_type TEXT NOT NULL DEFAULT 'Python Programming', -- e.g., 'Python Programming', 'Social Policy'
        assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL DEFAULT 'assigned', -- 'assigned', 'submitted', 'completed'
        llm_used_to_generate TEXT -- 'Gemini' or 'Ollama'
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER, -- Optional, for general questions, can be NULL
        submitted_content TEXT NOT NULL, -- Renamed from submitted_code
        submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ai_feedback TEXT,
        score INTEGER, -- 1-10 scale
        reviewed_by_llm TEXT, -- 'Gemini' or 'Ollama'
        submission_type TEXT NOT NULL, -- 'Python', 'Social Policy Article', 'Social Policy Question', 'Other'
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    # New table for settings like API keys
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_setting(key, value):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_setting(key):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_book_to_db(title, author, summary, filename):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO books (title, author, summary, filename)
        VALUES (?, ?, ?, ?)
        """, (title, author, summary, filename))
        conn.commit()
        st.sidebar.success(f"📖 Book '{title}' added to DB.")
    except sqlite3.IntegrityError:
        st.sidebar.warning(f"⚠️ Book with filename '{filename}' already exists in DB.")
    except Exception as e:
        st.sidebar.error(f"❌ Error adding book to DB: {e}")
    finally:
        conn.close()

def search_books_db(query_term):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, title, author, summary, filename
    FROM books
    WHERE title LIKE ? OR summary LIKE ?
    """, (f'%{query_term}%', f'%{query_term}%'))
    results = cursor.fetchall()
    conn.close()
    return results

def add_task_to_db(description, llm_used, task_type='Python Programming'):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO tasks (task_description, llm_used_to_generate, task_type)
    VALUES (?, ?, ?)
    """, (description, llm_used, task_type))
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return task_id

def get_active_tasks(task_type='Python Programming'):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch tasks that are 'assigned' or 'submitted' but not yet 'completed'
    cursor.execute("SELECT task_id, task_description, assigned_date, status FROM tasks WHERE status IN ('assigned', 'submitted') AND task_type = ? ORDER BY assigned_date DESC", (task_type,))
    tasks = cursor.fetchall()
    conn.close()
    return [{"task_id": t[0], "description": t[1], "assigned_date": t[2], "status": t[3]} for t in tasks]

def add_submission_to_db(task_id, submitted_content, ai_feedback, reviewed_by_llm, submission_type, score=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO submissions (task_id, submitted_content, ai_feedback, score, reviewed_by_llm, submission_type)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (task_id, submitted_content, ai_feedback, score, reviewed_by_llm, submission_type))
    submission_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return submission_id

def update_task_status(task_id, new_status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (new_status, task_id))
    conn.commit()
    conn.close()

def get_all_submissions(submission_type=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if submission_type:
        cursor.execute("""
        SELECT s.submission_id, COALESCE(t.task_description, 'N/A') as task_description, s.submitted_content, s.submission_date, s.ai_feedback, s.score, s.reviewed_by_llm, s.submission_type
        FROM submissions s
        LEFT JOIN tasks t ON s.task_id = t.task_id
        WHERE s.submission_type = ?
        ORDER BY s.submission_date DESC
        """, (submission_type,))
    else:
        cursor.execute("""
        SELECT s.submission_id, COALESCE(t.task_description, 'N/A') as task_description, s.submitted_content, s.submission_date, s.ai_feedback, s.score, s.reviewed_by_llm, s.submission_type
        FROM submissions s
        LEFT JOIN tasks t ON s.task_id = t.task_id
        ORDER BY s.submission_date DESC
        """)
    submissions = cursor.fetchall()
    conn.close()
    return submissions

def get_all_tasks(task_type=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if task_type:
        cursor.execute("SELECT task_id, task_description, assigned_date, status, task_type FROM tasks WHERE task_type = ? ORDER BY assigned_date DESC", (task_type,))
    else:
        cursor.execute("SELECT task_id, task_description, assigned_date, status, task_type FROM tasks ORDER BY assigned_date DESC")
    tasks = cursor.fetchall()
    conn.close()
    return tasks


# Initialize DB on first run
init_db()

# --- LLM HELPER FUNCTIONS ---

# Initialize session state for LLM configurations
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = load_setting("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "") # Load from DB first, then env var
if 'gemini_model_name' not in st.session_state:
    st.session_state.gemini_model_name = 'gemini-1.5-flash-latest' # Default Gemini model
if 'ollama_model_name' not in st.session_state:
    st.session_state.ollama_model_name = DEFAULT_OLLAMA_MODEL
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'gemini_enabled' not in st.session_state:
    st.session_state.gemini_enabled = False
if 'available_gemini_models' not in st.session_state:
    st.session_state.available_gemini_models = []
if 'available_ollama_models' not in st.session_state:
    st.session_state.available_ollama_models = []

# New session states for task management
if 'current_python_task_id' not in st.session_state:
    st.session_state.current_python_task_id = None
if 'current_python_task_description' not in st.session_state:
    st.session_state.current_python_task_description = ""
if 'python_feedback' not in st.session_state:
    st.session_state.python_feedback = ""


def configure_gemini_client():
    """Configures the Gemini client if an API key is available."""
    if st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            # Test with a simple model listing to ensure key is valid
            models = genai.list_models() # Check if API key is valid
            st.session_state.gemini_client = genai.GenerativeModel(st.session_state.gemini_model_name)
            st.session_state.gemini_enabled = True
            fetch_available_gemini_models() # Fetch models after successful configuration
            return True
        except Exception as e:
            st.sidebar.error(f"🔑 Gemini Config Error: {str(e)[:100]}...")
            st.session_state.gemini_enabled = False
            st.session_state.gemini_client = None
            st.session_state.available_gemini_models = []
            return False
    else:
        st.session_state.gemini_enabled = False
        st.session_state.gemini_client = None
        st.session_state.available_gemini_models = []
        return False

def fetch_available_gemini_models():
    """Fetches and filters available Gemini models."""
    if st.session_state.gemini_enabled and st.session_state.gemini_api_key: # Ensure key is set before trying
        try:
            models_list = []
            for m in genai.list_models():
                # Filter for models that support 'generateContent' and are typical text models
                if 'generateContent' in m.supported_generation_methods and ("gemini" in m.name):
                    model_id = m.name.split('/')[-1]
                    models_list.append(model_id)
            st.session_state.available_gemini_models = sorted(list(set(models_list)))
            if not st.session_state.available_gemini_models:
                 st.sidebar.warning("No compatible Gemini models found, or API key issue.")
            # Ensure current model is in the list, if not, reset to default or first available
            if st.session_state.gemini_model_name not in st.session_state.available_gemini_models:
                st.session_state.gemini_model_name = 'gemini-1.5-flash-latest' if 'gemini-1.5-flash-latest' in st.session_state.available_gemini_models else (st.session_state.available_gemini_models[0] if st.session_state.available_gemini_models else None)

        except Exception as e:
            st.sidebar.error(f"Failed to fetch Gemini models: {str(e)[:100]}...")
            st.session_state.available_gemini_models = []
    else:
        st.session_state.available_gemini_models = []


def fetch_available_ollama_models():
    """Fetches locally available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        st.session_state.available_ollama_models = sorted([model['name'] for model in models_data.get('models', [])])
        if not st.session_state.available_ollama_models:
            st.sidebar.warning("No Ollama models found. Is Ollama running and models pulled?")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("Ollama connection error. Is it running?")
        st.session_state.available_ollama_models = []
    except Exception as e:
        st.sidebar.error(f"Error fetching Ollama models: {e}")
        st.session_state.available_ollama_models = []


# Initial fetch of Ollama models and Gemini configuration
if not st.session_state.available_ollama_models: # Fetch only if not already populated
    fetch_available_ollama_models()
# Configure Gemini if key exists (either from DB or env var)
if not st.session_state.gemini_enabled and st.session_state.gemini_api_key:
    configure_gemini_client()


def get_ollama_response(prompt_text):
    if not OLLAMA_BASE_URL: return "Ollama URL not configured."
    if not st.session_state.ollama_model_name: return "No Ollama model selected."
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": st.session_state.ollama_model_name, "prompt": prompt_text, "stream": False},
            timeout=1800 # Increased timeout for potentially complex tasks
        )
        response.raise_for_status()
        return json.loads(response.text)["response"]
    except Exception as e:
        st.error(f"Ollama error ({st.session_state.ollama_model_name}): {e}")
        return f"Error with Ollama: {e}"

def get_gemini_response(prompt_parts):
    if not st.session_state.gemini_enabled or not st.session_state.gemini_client:
        return "Gemini is not enabled or configured. Please check API key and model selection."
    try:
        # Re-initialize client with the currently selected model name, just in case it changed
        current_gemini_model_client = genai.GenerativeModel(st.session_state.gemini_model_name)
        response = current_gemini_model_client.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error ({st.session_state.gemini_model_name}): {e}")
        return f"Error with Gemini API: {e}"


def parse_and_extract_score(ai_response):
    """Parses AI response to extract a score, if present."""
    match = re.search(r"SCORE:\s*(\d+)/10", ai_response)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def list_files_in_directory(directory_path):
    try:
        return [f for f in os.listdir(directory_path) if f.endswith((".txt", ".md"))]
    except Exception:
        return []

def read_file_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="AI Learning System 🚀")
st.title("🎓 Comprehensive AI Learning System")
st.caption("Your AI mentor for Python, Social Policy, and more. Powered by Gemini and Ollama.")

# Initialize session state for UI elements if not already done
ui_defaults = {
    'social_policy_response': "",
    'social_policy_topic': "",
    'article_analysis_feedback': "",
    'other_courses_suggestion': ""
}
for key, value in ui_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if 'selected_primary_llm' not in st.session_state: # For overall LLM preference
    # Set initial preference based on Gemini availability, then Ollama
    if st.session_state.gemini_enabled:
        st.session_state.selected_primary_llm = "Gemini"
    elif st.session_state.available_ollama_models:
        st.session_state.selected_primary_llm = "Ollama"
    else:
        st.session_state.selected_primary_llm = None # No LLMs available


# --- SIDEBAR ---
st.sidebar.header("⚙️ LLM Configuration")

# Gemini API Key Input
st.sidebar.subheader("🔑 Google Gemini Setup")
user_gemini_api_key = st.sidebar.text_input(
    "Enter Gemini API Key:",
    type="password",
    value=st.session_state.gemini_api_key,
    help="Your API key is not stored persistently by this app after session ends. It will be stored in your local SQLite database."
)
if st.sidebar.button("Apply Gemini API Key", key="apply_gemini_key"):
    st.session_state.gemini_api_key = user_gemini_api_key
    save_setting("gemini_api_key", user_gemini_api_key) # Save to DB
    if configure_gemini_client():
        st.sidebar.success("Gemini API Key accepted & configured! ✨ (Saved to DB)")
        # Update primary LLM preference if Gemini just became available
        if st.session_state.selected_primary_llm is None or st.session_state.selected_primary_llm == "Ollama":
             st.session_state.selected_primary_llm = "Gemini"
    else:
        st.sidebar.error("Failed to configure Gemini with the provided key.")


if st.session_state.gemini_enabled and st.session_state.available_gemini_models:
    st.session_state.gemini_model_name = st.sidebar.selectbox(
        "Select Gemini Model:",
        options=st.session_state.available_gemini_models,
        index=st.session_state.available_gemini_models.index(st.session_state.gemini_model_name) if st.session_state.gemini_model_name in st.session_state.available_gemini_models else 0,
        key="gemini_model_selector"
    )
elif st.session_state.gemini_api_key and not st.session_state.gemini_enabled:
    st.sidebar.warning("Gemini API Key provided, but client not configured. Click 'Apply Key'.")
else:
    st.sidebar.info("Enter a Gemini API Key to enable Gemini models.")


st.sidebar.markdown("---")
st.sidebar.subheader("🦙 Ollama (Local) Setup")
if st.session_state.available_ollama_models:
    # Ensure default is in list, or pick first one
    default_ollama_idx = 0
    if st.session_state.ollama_model_name in st.session_state.available_ollama_models:
        default_ollama_idx = st.session_state.available_ollama_models.index(st.session_state.ollama_model_name)
    elif DEFAULT_OLLAMA_MODEL in st.session_state.available_ollama_models:
         default_ollama_idx = st.session_state.available_ollama_models.index(DEFAULT_OLLAMA_MODEL)


    st.session_state.ollama_model_name = st.sidebar.selectbox(
        "Select Ollama Model:",
        options=st.session_state.available_ollama_models,
        index=default_ollama_idx,
        key="ollama_model_selector"
    )
    st.sidebar.caption(f"Using Ollama from: {OLLAMA_BASE_URL}")
else:
    st.sidebar.warning("No Ollama models detected. Ensure Ollama is running and models are pulled (e.g., `ollama pull gemma:latest`).")
if st.sidebar.button("Refresh Ollama Models List", key="refresh_ollama_list"):
    fetch_available_ollama_models()
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.subheader("💡 Preferred LLM for Tasks")
# Determine available primary LLMs based on successful configuration
primary_llm_options = []
if st.session_state.gemini_enabled:
    primary_llm_options.append("Gemini")
if st.session_state.available_ollama_models: # Check if Ollama has models
    primary_llm_options.append("Ollama")

if primary_llm_options:
    # Ensure the selected primary LLM is valid, otherwise default
    current_primary_idx = 0
    if st.session_state.selected_primary_llm in primary_llm_options:
        current_primary_idx = primary_llm_options.index(st.session_state.selected_primary_llm)
    elif primary_llm_options: # If current selection is invalid, pick the first available
        st.session_state.selected_primary_llm = primary_llm_options[0]

    st.session_state.selected_primary_llm = st.sidebar.radio(
        "Choose primary LLM for generating responses:",
        options=primary_llm_options,
        index=current_primary_idx, # Use the potentially corrected index
        key="primary_llm_selector"
    )
else:
    st.sidebar.error("No LLMs available. Configure Gemini API Key or ensure Ollama is running with models.")
    st.session_state.selected_primary_llm = None


st.sidebar.markdown("---")
st.sidebar.subheader("📚 Add Book to Database")
with st.sidebar.expander("➕ New Book Form", expanded=False):
    new_book_title = st.text_input("Book Title", key="new_book_title_v3")
    new_book_author = st.text_input("Author (Optional)", key="new_book_author_v3")
    new_book_summary = st.text_area("Brief Summary", key="new_book_summary_v3")
    available_book_files_for_db = list_files_in_directory(BOOKS_DIR)
    new_book_filename = st.selectbox("Select Book File (from data/books/)", ["None"] + available_book_files_for_db, key="new_book_filename_v3")

    if st.button("Add Book to DB", key="add_book_db_btn_v3"):
        if new_book_title and new_book_summary and new_book_filename != "None":
            add_book_to_db(new_book_title, new_book_author, new_book_summary, new_book_filename)
        else:
            st.sidebar.error("Title, Summary, and Filename are required.")

st.sidebar.markdown("---")
st.sidebar.info(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Local)")


# --- MAIN TABS ---
tab_python, tab_social_policy, tab_other_courses, tab_dashboard = st.tabs([
    "🐍 Python Programming Tutor",
    "🏛️ Social Policy Analyst Prep",
    "🌱 Related Learning Paths",
    "📊 Dashboard"
])

# --- PYTHON PROGRAMMING TUTOR TAB ---
with tab_python:
    st.header("🐍 Python Programming Tutor")
    st.markdown("""
    - **Get New Tasks:** Generate and manage multiple Python programming challenges.
    - **Submit Your Code:** Get feedback, corrections, and explanations tailored to the specific task you're working on.
    - **Track Progress:** All tasks and submissions are saved.
    """)

    st.subheader("📚 Assigned Python Tasks")
    assigned_tasks = get_active_tasks(task_type='Python Programming')
    
    # Handle initial task generation if no tasks exist
    if not assigned_tasks:
        st.info("No active Python tasks found. Let's generate your first one!")
        if st.session_state.selected_primary_llm:
            with st.spinner(f"Generating your initial Python task ({st.session_state.selected_primary_llm})..."):
                llm_to_use = st.session_state.selected_primary_llm
                prompt = """You are an AI Python programming tutor.
                Please generate a new, unique Python programming task suitable for a beginner to intermediate learner.
                The task should be well-defined and solvable within a reasonable amount of time (e.g., 15-60 minutes).
                Provide only the task description.
                Example: 'Write a Python function that takes a string and returns the number of vowels in it.'
                """
                if llm_to_use == "Gemini":
                    task_response = get_gemini_response([prompt])
                else: # Ollama
                    task_response = get_ollama_response(prompt)
                
                if task_response and not task_response.startswith("Error"):
                    task_id = add_task_to_db(task_response, llm_to_use, task_type='Python Programming')
                    st.session_state.current_python_task_id = task_id
                    st.session_state.current_python_task_description = task_response
                    st.success("Your first task has been generated!")
                    st.rerun() # Rerun to update the task display
                else:
                    st.error(f"Failed to generate initial task: {task_response}")
        else:
            st.warning("Please configure at least one LLM (Gemini or Ollama) to generate tasks.")
    else: # If tasks exist, allow selection
        task_options = {f"Task {t['task_id']}: {t['description'][:70]}... (Status: {t['status']})" : t['task_id'] for t in assigned_tasks}
        
        # Ensure current_python_task_id is valid, or set to the first task
        if st.session_state.current_python_task_id not in task_options.values():
            st.session_state.current_python_task_id = assigned_tasks[0]['task_id']
            st.session_state.current_python_task_description = assigned_tasks[0]['description']

        selected_task_display = st.selectbox(
            "Select an active task to work on or review:",
            options=list(task_options.keys()),
            format_func=lambda x: x,
            key="selected_task_for_work",
            index=list(task_options.values()).index(st.session_state.current_python_task_id)
        )
        st.session_state.current_python_task_id = task_options[selected_task_display]
        # Update description based on selected task
        for t in assigned_tasks:
            if t['task_id'] == st.session_state.current_python_task_id:
                st.session_state.current_python_task_description = t['description']
                break

        st.info(f"**CURRENT TASK:** {st.session_state.current_python_task_description}")
        st.markdown("---")
        # Display past submissions for the current task
        task_submissions = [s for s in get_all_submissions(submission_type='Python') if s[1] == st.session_state.current_python_task_description]
        if task_submissions:
            st.subheader(f"Previous Submissions for Task {st.session_state.current_python_task_id}")
            for i, sub in enumerate(task_submissions):
                with st.expander(f"Submission {i+1} on {sub[3]} (Score: {sub[5] if sub[5] else 'N/A'})"):
                    st.code(sub[2], language="python")
                    st.markdown("**AI Feedback:**")
                    st.markdown(sub[4])
                    if sub[5]:
                        st.success(f"**Score: {sub[5]}/10**")
                    else:
                        st.warning("No score extracted from feedback.")
        st.markdown("---")
    

    st.subheader("Generate New Python Task")
    if st.button("✨ Generate New Python Task", key="new_python_task_btn_v3"):
        if st.session_state.selected_primary_llm:
            prompt = """You are an AI Python programming tutor.
            Please generate a new, unique Python programming task suitable for a beginner to intermediate learner.
            The task should be well-defined and solvable within a reasonable amount of time (e.g., 15-60 minutes).
            Provide only the task description.
            Example: 'Write a Python function that takes a string and returns the number of vowels in it.'
            """
            with st.spinner(f"Generating new Python task ({st.session_state.selected_primary_llm})..."):
                llm_to_use = st.session_state.selected_primary_llm
                if llm_to_use == "Gemini":
                    task_response = get_gemini_response([prompt])
                else: # Ollama
                    task_response = get_ollama_response(prompt)

                if task_response and not task_response.startswith("Error"):
                    task_id = add_task_to_db(task_response, llm_to_use, task_type='Python Programming')
                    st.success(f"New task added: Task ID {task_id}")
                    st.rerun() # Rerun to refresh the task list
                else:
                    st.error(f"Failed to generate task: {task_response}")
        else:
            st.warning("Please configure at least one LLM to generate tasks.")

    st.subheader("✍️ Submit Your Python Code or Ask a Question")
    user_python_input = st.text_area("Paste your Python code here, or type your Python-related question:", height=200, key="python_code_input_v3")

    col_code_btn_py, col_mark_complete = st.columns(2)

    with col_code_btn_py:
        if st.button("🔍 Get Code Review / Explain", key="review_python_code_btn_v3"):
            extracted_score = None # Initialize to None to fix NameError
            if user_python_input and st.session_state.current_python_task_id:
                prompt = f"""You are an expert Python programming tutor.
                The student was given the following task:
                TASK: {st.session_state.current_python_task_description}

                They have submitted the following Python code or question:
                ```python
                {user_python_input}
                ```
                Your tasks are to:
                1.  **Crucially, evaluate the submitted code against the specific requirements of the TASK provided.**
                2.  If it's code:
                    a.  Identify any errors (syntax, logical) in the context of the task.
                    b.  Provide a corrected version of the code, explaining the changes clearly and how they address the task requirements or fix errors.
                    c.  Explain the underlying Python concepts related to the errors or the code's purpose.
                    d.  Offer suggestions for improvement or alternative approaches, especially in the context of the task.
                    e.  **Grade the code based on the following indicators, provide a brief comment for each, and then an overall score:**
                        -   **Correctness (1-10):** Does the code produce the correct output for the task?
                        -   **Adherence to Task (1-10):** How well does the code meet all requirements of the task?
                        -   **Readability (1-10):** Is the code well-organized, commented, and easy to understand?
                        -   **Efficiency (1-10):** Is the code reasonably efficient for the given task?
                        -   **Error Handling (1-10):** Does the code handle potential errors gracefully (if applicable to the task)?
                3.  If it's a question:
                    a.  Answer the question clearly and comprehensively.
                    b.  Provide examples where appropriate.
                    c.  **Do NOT provide a numerical score for questions.**
                4.  **For code submissions, provide a single overall score at the very beginning of your response on a scale of 1-10, indicating how well the code fulfills the task and exhibits good practices. Format it exactly like: 'SCORE: X/10' (e.g., 'SCORE: 8/10')**
                5.  Maintain a supportive and encouraging tone.
                6.  Structure your response clearly using Markdown. For code blocks, specify the language as python.
                """
                with st.spinner(f"Python Tutor ({st.session_state.selected_primary_llm}) is thinking..."):
                    llm_used_for_review = st.session_state.selected_primary_llm
                    if llm_used_for_review == "Gemini":
                        ai_response = get_gemini_response([prompt])
                    else: # Ollama
                        ai_response = get_ollama_response(prompt)

                    st.session_state.python_feedback = ai_response
                    extracted_score = parse_and_extract_score(ai_response)
                    
                    if st.session_state.current_python_task_id:
                        add_submission_to_db(
                            st.session_state.current_python_task_id,
                            user_python_input,
                            ai_response,
                            llm_used_for_review,
                            'Python', # submission_type
                            extracted_score
                        )
                        st.success("Your submission has been recorded!")
                    else:
                        st.warning("No active task selected. Feedback not saved to a specific task.")

            elif not user_python_input:
                st.session_state.python_feedback = "Please enter some Python code or a question first."
            else:
                st.session_state.python_feedback = "Please select a task from 'Assigned Python Tasks' above before submitting code."
    
    with col_mark_complete:
        if st.session_state.current_python_task_id and st.button("✅ Mark Task as Completed", key="mark_task_complete_btn"):
            update_task_status(st.session_state.current_python_task_id, 'completed')
            st.success(f"Task {st.session_state.current_python_task_id} marked as completed! 🎉")
            st.session_state.current_python_task_id = None # Clear current task selection
            st.session_state.current_python_task_description = ""
            st.session_state.python_feedback = ""
            st.rerun() # Rerun to update the task list


    if st.session_state.python_feedback:
        st.subheader("💡 Python Tutor Feedback")
        # extracted_score might be None if no code was submitted or score not parsed
        if extracted_score is not None:
            st.markdown(f"### Overall Score: {extracted_score}/10")
        st.markdown(st.session_state.python_feedback)


# --- SOCIAL POLICY ANALYST PREP TAB ---
with tab_social_policy:
    st.header("🏛️ Social Policy Analyst Prep")
    st.markdown("""
    Develop your skills as a Social Policy Analyst.
    - **Explore Topics & Case Studies:** Get suggestions for areas of study.
    - **Ask Questions:** Query theories, concepts, or current issues, optionally referencing your saved materials.
    - **Analyze Your Writing:** Submit your social policy articles for AI-powered feedback.
    """)

    st.subheader("📖 Explore Topics & Case Studies")
    if st.button("🌍 Suggest a Social Policy Topic or Case Study", key="suggest_sp_topic_btn_v3"):
        prompt = """You are an AI assistant mentoring an aspiring Social Policy Analyst, particularly with an interest in Nigeria and developing countries.
        Suggest a relevant and engaging social policy topic for research OR outline a brief case study scenario.
        Make it specific enough to be actionable for study.
        For a topic, suggest what aspects to focus on.
        For a case study, briefly describe the situation and key questions to consider.
        Example Topic: 'The Impact of Youth Empowerment Schemes on Unemployment Rates in Urban Nigeria: A Critical Analysis of [Specific Scheme]. Focus on methodology, data sources, and policy recommendations.'
        Example Case Study: 'A rapidly growing city in a developing nation is facing a severe housing crisis due to rural-urban migration. The government proposes a public-private partnership for affordable housing. What are the potential benefits, challenges, key stakeholders, and ethical considerations for a social policy analyst to examine?'
        """
        with st.spinner(f"Generating topic/case study ({st.session_state.selected_primary_llm})..."):
            if st.session_state.selected_primary_llm == "Gemini":
                st.session_state.social_policy_topic = get_gemini_response([prompt])
            else:
                st.session_state.social_policy_topic = get_ollama_response(prompt)

    if st.session_state.social_policy_topic:
        st.info(f"TOPIC/CASE STUDY IDEA:\n{st.session_state.social_policy_topic}")

    st.subheader("❓ Ask Your Social Policy Question")
    user_sp_question = st.text_area("Type your question about social policy:", height=100, key="sp_question_v3")

    st.markdown("**Optional: Provide Context for Your Question**")
    col_db_search, col_file_select = st.columns(2)

    selected_db_book_summary = ""
    selected_db_book_title = ""

    with col_db_search:
        st.markdown("###### 📚 From Your Book Database")
        db_search_term = st.text_input("Search book summaries (titles/summaries in DB):", key="db_search_term_v3")
        if db_search_term:
            db_book_results = search_books_db(db_search_term)
            if db_book_results:
                st.write("Found in your DB (select one):")
                # Use radio buttons for single selection from DB results
                db_selection_options = {f"{book[1]} (by {book[2] or 'N/A'})": book for book in db_book_results}
                selected_book_display_name = st.radio("Select book summary to include:", ["None"] + list(db_selection_options.keys()), key="db_book_radio_select")

                if selected_book_display_name != "None":
                    selected_book_data = db_selection_options[selected_book_display_name]
                    selected_db_book_summary = selected_book_data[3]
                    selected_db_book_title = selected_book_data[1]
                    with st.expander("View selected summary", expanded=False):
                        st.caption(f"{selected_db_book_summary[:300]}...")
            else:
                st.caption("No matching books found in your database for that term.")

    selected_article_content = ""
    selected_full_book_content = ""

    with col_file_select:
        st.markdown("###### 📄 From Your Local Files")
        available_articles = list_files_in_directory(ARTICLES_DIR)
        selected_article_file = st.selectbox("Select an article file (from data/articles/):", ["None"] + available_articles, key="article_select_v3")
        if selected_article_file != "None":
            content = read_file_content(os.path.join(ARTICLES_DIR, selected_article_file))
            if content:
                selected_article_content = content
                with st.expander("Preview selected article", expanded=False): st.text(content[:500] + "...")

        available_book_files = list_files_in_directory(BOOKS_DIR)
        selected_book_file_full = st.selectbox("Select a full book text file (from data/books/):", ["None"] + available_book_files, key="book_select_full_v3")
        if selected_book_file_full != "None":
            content = read_file_content(os.path.join(BOOKS_DIR, selected_book_file_full))
            if content:
                selected_full_book_content = content
                with st.expander("Preview selected full book text", expanded=False): st.text(content[:500] + "...")


    if st.button("💬 Get Social Policy Insight", key="get_sp_insight_btn_v3"):
        extracted_score = None # Initialize
        if user_sp_question:
            context_text = "The user is asking a question about social policy. "
            if selected_db_book_summary:
                context_text += f"\n\nConsider this summary from the user's book database for context (Book Title: '{selected_db_book_title}'):\n'{selected_db_book_summary}'\n"
            if selected_article_content:
                context_text += f"\n\nConsider this content from an article provided by the user:\n'{selected_article_content[:2000]}'\n"
            if selected_full_book_content:
                context_text += f"\n\nConsider this content from a full book text provided by the user:\n'{selected_full_book_content[:2000]}'\n"

            prompt = f"""You are an expert Social Policy AI mentor, drawing upon a broad understanding of theories, global practices, and specific contexts like Nigeria.
            A student has the following question: "{user_sp_question}"

            {context_text if (selected_db_book_summary or selected_article_content or selected_full_book_content) else "No specific local documents were provided by the user for this query."}

            Please provide a comprehensive, insightful, and well-structured answer.
            If context from provided documents/summaries was used, briefly mention how it informed your answer.
            Aim to 'draw from experience' by synthesizing information, considering multiple perspectives (economic, social, political, ethical), and discussing potential implications.
            **Do NOT provide a numerical score for questions.**
            Structure your response clearly using Markdown.
            """
            with st.spinner(f"Social Policy Mentor ({st.session_state.selected_primary_llm}) is thinking..."):
                llm_used_for_review = st.session_state.selected_primary_llm
                if llm_used_for_review == "Gemini":
                    st.session_state.social_policy_response = get_gemini_response([prompt])
                else:
                    st.session_state.social_policy_response = get_ollama_response(prompt)
            
            # For questions, no score is extracted, but we still log the interaction
            add_submission_to_db(
                None, # No task_id for a general question
                user_sp_question,
                st.session_state.social_policy_response,
                llm_used_for_review,
                'Social Policy Question', # submission_type
                None # No score for questions
            )
            st.success("Your question and the AI's response have been recorded.")
        else:
            st.session_state.social_policy_response = "Please type a question first."

    if st.session_state.social_policy_response:
        st.subheader("📖 Social Policy Mentor's Response")
        st.markdown(st.session_state.social_policy_response)

    st.markdown("---")
    st.subheader("✍️ Submit Your Social Policy Article for Analysis")
    article_submission_method = st.radio("How would you like to submit your article?", ("Paste Text", "Upload .txt File"), key="article_method_v3", horizontal=True)

    submitted_article_text = ""
    if article_submission_method == "Paste Text":
        submitted_article_text = st.text_area("Paste your article text here:", height=300, key="pasted_article_v3")
    else:
        uploaded_file = st.file_uploader("Upload your .txt article file:", type=["txt"], key="uploaded_article_v3")
        if uploaded_file is not None:
            try:
                submitted_article_text = uploaded_file.read().decode("utf-8")
                st.success("File uploaded successfully. Preview (first 500 chars):")
                st.text(submitted_article_text[:500]+"...")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    if st.button("🔬 Analyze My Article", key="analyze_article_btn_v3"):
        extracted_score = None # Initialize
        if submitted_article_text:
            prompt = f"""You are an experienced Social Policy academic supervisor.
            A student has submitted the following article for review:
            --- BEGIN ARTICLE ---
            {submitted_article_text}
            --- END ARTICLE ---

            Please provide a constructive and detailed analysis of this article. Focus on:
            1.  **Clarity of Thesis/Argument (1-10):** Is the main argument clear, well-defined, and consistently maintained?
            2.  **Use of Evidence (1-10):** Is evidence used effectively to support claims? Is it relevant, sufficient, and properly integrated? (Acknowledge you don't have external access to verify facts but can assess how evidence is presented).
            3.  **Structure and Coherence (1-10):** Is the article well-organized? Do sections and paragraphs flow logically?
            4.  **Understanding of Social Policy Concepts (1-10):** Does the author demonstrate a strong grasp of relevant social policy theories, frameworks, or concepts?
            5.  **Critical Analysis (1-10):** Does the article engage critically with the topic, considering different perspectives, limitations, or complexities?
            6.  **Writing Style & Clarity (1-10):** Is the language clear, concise, academic in tone, and free from major grammatical errors? Are there areas for improvement in expression?
            7.  **Specific Recommendations:** Offer 2-3 actionable recommendations for improvement.

            **Provide a single overall score at the very beginning of your response on a scale of 1-10. Format it exactly like: 'SCORE: X/10' (e.g., 'SCORE: 7/10')**
            Be thorough and supportive. Structure your feedback using Markdown, detailing comments for each indicator.
            """
            with st.spinner(f"Analyzing article ({st.session_state.selected_primary_llm})... This may take a moment."):
                llm_used_for_review = st.session_state.selected_primary_llm
                if llm_used_for_review == "Gemini":
                    ai_response = get_gemini_response([prompt])
                else:
                    ai_response = get_ollama_response(prompt)
                
                st.session_state.article_analysis_feedback = ai_response
                extracted_score = parse_and_extract_score(ai_response)

                add_submission_to_db(
                    None, # No specific task_id for general article analysis
                    submitted_article_text,
                    ai_response,
                    llm_used_for_review,
                    'Social Policy Article', # submission_type
                    extracted_score
                )
                st.success("Your article analysis and feedback have been recorded!")
        else:
            st.session_state.article_analysis_feedback = "Please paste or upload an article to analyze."

    if st.session_state.article_analysis_feedback:
        st.subheader("🔍 Article Analysis Feedback")
        if extracted_score is not None:
            st.markdown(f"### Overall Score: {extracted_score}/10")
        st.markdown(st.session_state.article_analysis_feedback)


# --- RELATED LEARNING PATHS TAB ---
with tab_other_courses:
    st.header("🌱 Related Learning Paths & Skills")
    st.markdown("""
    Discover other courses, skills, or fields of study that can complement your journey in Python programming and Social Policy Analysis.
    """)

    if st.button("💡 Suggest Complementary Learning Paths", key="suggest_courses_btn_v3"):
        prompt = """You are an AI academic and career advisor.
        A user is currently learning Python programming and preparing to be a Social Policy Analyst, with a likely focus on Nigeria or developing countries.
        Suggest 3-5 complementary fields of study, specific skills (technical or soft), or types of courses that would be beneficial for their career goals.
        For each suggestion, briefly explain its relevance.
        Example: 'Data Analysis & Visualization: Skills in tools like R or Python libraries (Pandas, Matplotlib, Seaborn) are crucial for analyzing social data and presenting findings effectively for policy recommendations.'
        """
        with st.spinner(f"Thinking about related paths ({st.session_state.selected_primary_llm})..."):
            if st.session_state.selected_primary_llm == "Gemini":
                st.session_state.other_courses_suggestion = get_gemini_response([prompt])
            else:
                st.session_state.other_courses_suggestion = get_ollama_response(prompt)

    if st.session_state.other_courses_suggestion:
        st.markdown(st.session_state.other_courses_suggestion)

# --- DASHBOARD TAB ---
with tab_dashboard:
    st.header("📊 Your Learning Dashboard")
    st.markdown("Track your progress, review past submissions, and see your scores.")

    all_submissions = get_all_submissions()
    all_tasks_from_db = get_all_tasks() # Get all tasks (Python and potential future Social Policy tasks)

    # Filter tasks by type for summary
    python_tasks = [t for t in all_tasks_from_db if t[4] == 'Python Programming']
    
    total_python_tasks = len(python_tasks)
    completed_python_tasks = sum(1 for task in python_tasks if task[3] == 'completed')
    
    python_submissions = [s for s in all_submissions if s[7] == 'Python']
    social_policy_article_submissions = [s for s in all_submissions if s[7] == 'Social Policy Article']
    social_policy_question_submissions = [s for s in all_submissions if s[7] == 'Social Policy Question']


    st.subheader("Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Python Tasks Assigned", total_python_tasks)
    col2.metric("Python Submissions", len(python_submissions))
    col3.metric("Python Tasks Completed", completed_python_tasks)
    col4.metric("Social Policy Articles Submitted", len(social_policy_article_submissions))
    col5.metric("Social Policy Questions Asked", len(social_policy_question_submissions))


    # Prepare data for scoring and charts
    scored_submissions = [s for s in all_submissions if s[5] is not None] # Filter for submissions with a score

    if scored_submissions:
        df_submissions = pd.DataFrame(scored_submissions, columns=[
            "Submission ID", "Task Description", "Submitted Content", "Submission Date",
            "AI Feedback", "Score", "Reviewed By LLM", "Submission Type"
        ])
        df_submissions["Submission Date"] = pd.to_datetime(df_submissions["Submission Date"])

        # Calculate average score for submissions that have a score
        average_score = df_submissions["Score"].mean()
        st.metric("Overall Average Score", f"{average_score:.2f}/10")

        st.subheader("Score Distribution")
        score_counts = df_submissions["Score"].value_counts().sort_index()
        st.bar_chart(score_counts)
        
        st.subheader("All Your Submissions")
        
        # Add a filter for submission type
        submission_types_for_filter = ["All"] + list(df_submissions["Submission Type"].unique())
        selected_submission_type = st.selectbox("Filter Submissions by Type:", submission_types_for_filter, key="dashboard_submission_type_filter")

        display_df = df_submissions
        if selected_submission_type != "All":
            display_df = df_submissions[df_submissions["Submission Type"] == selected_submission_type]

        # Display submissions without the potentially long code and feedback columns for summary table
        st.dataframe(display_df[["Submission ID", "Task Description", "Submission Type", "Submission Date", "Score", "Reviewed By LLM"]])

        st.subheader("Detailed Submission History")
        if display_df.empty:
            st.info(f"No submissions found for type: {selected_submission_type}.")
        else:
            for i, row in display_df.iterrows():
                with st.expander(f"Submission {row['Submission ID']} - Type: {row['Submission Type']} - Task: {row['Task Description'][:70]}... (Score: {row['Score'] if row['Score'] else 'N/A'}) on {row['Submission Date'].strftime('%Y-%m-%d %H:%M')}"):
                    st.markdown(f"**Submission Type:** {row['Submission Type']}")
                    st.markdown(f"**Associated Task:** {row['Task Description']}")
                    st.markdown(f"**Submitted Content:**")
                    # Use st.code for Python, st.text for others for better formatting
                    if row['Submission Type'] == 'Python':
                        st.code(row['Submitted Content'], language="python")
                    else:
                        st.text(row['Submitted Content'])
                    st.markdown(f"**AI Feedback:**")
                    st.markdown(row['AI Feedback'])
                    if row['Score']:
                        st.success(f"**Extracted Score: {row['Score']}/10**")
                    else:
                        st.warning("No numerical score was extracted from this feedback.")
                    st.markdown(f"*(Reviewed by: {row['Reviewed By LLM']})*")
    else:
        st.info("No scored submissions yet. Complete some tasks or analyze articles to see your progress!")
import streamlit as st
import requests
import re
import json
from pathlib import Path

# Configure page
st.set_page_config(page_title="SRT Subtitle Translator", layout="wide")

# Title
st.title("🎬 Movie Subtitle Translator (English to Malayalam)")

# API Key input
api_key = st.text_input("Enter your Gemini API Key:", type="password")

# File upload
uploaded_file = st.file_uploader("Upload SRT File", type=['srt'])

def parse_srt(content):
    """Parse SRT file content into structured data"""
    lines = content.strip().split('\n')
    subtitles = []
    i = 0
    
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
            
        # Read subtitle number
        try:
            sub_num = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
            
        # Read timestamp
        if i + 1 < len(lines):
            timestamp = lines[i + 1].strip()
        else:
            break
            
        # Read subtitle text (can be multiple lines)
        text_lines = []
        i += 2
        while i < len(lines) and lines[i].strip() and not re.match(r'^\d+$', lines[i].strip()):
            text_lines.append(lines[i].strip())
            i += 1
            
        text = '\n'.join(text_lines)
        
        subtitles.append({
            'number': sub_num,
            'timestamp': timestamp,
            'text': text
        })
    
    return subtitles

def format_selected_subtitles(subtitles, start, end=None):
    """Format selected subtitle range for display and API"""
    if end is None or end < start:
        end = start
    
    selected = []
    for sub in subtitles:
        if start <= sub['number'] <= end:
            selected.append(sub)
    
    # Format for display
    formatted_text = ""
    for sub in selected:
        formatted_text += f"{sub['number']}\n"
        formatted_text += f"{sub['timestamp']}\n"
        formatted_text += f"{sub['text']}\n\n"
    
    return selected, formatted_text.strip()

def translate_with_gemini(selected_subs, movie_name, full_context, api_key):
    """Translate subtitles using Gemini API via requests"""
    
    # Create context-aware prompt
    context_text = "\n".join([f"Line {s['number']}: {s['text']}" for s in full_context])
    
    translation_text = "\n\n".join([
        f"Line {s['number']} ({s['timestamp']}):\n{s['text']}" 
        for s in selected_subs
    ])
    
    # prompt = f"""You are a professional movie subtitle translator specializing in English to Malayalam translation.

# Movie: {movie_name}

# CONTEXT (surrounding subtitles for better understanding):
# {context_text}

# SUBTITLES TO TRANSLATE:
# {translation_text}

# Instructions:
# 1. Translate the subtitles from English to Malayalam
# 2. Preserve the essence, emotion, and tone of the original dialogue
# 3. Keep cultural context and movie context in mind
# 4. Maintain natural Malayalam dialogue flow
# 6. Consider the timestamps to understand scene pacing
# 7. Return ONLY the translated Malayalam text for each line number, maintaining the line breaks

# Format your response as:
# Line [number]:
# [Malayalam translation]

# Line [number]:
# [Malayalam translation]
# """
    prompt = f"""
You are an expert **movie subtitle translator** specializing in **English-to-Malayalam** translations for dubbed films.

🎬 Movie: {movie_name}

📖 CONTEXT (surrounding subtitles for reference):
{context_text}

🎯 SUBTITLES TO TRANSLATE (line {start_line} to {end_line}):
{translation_text}

---

### Translation Guidelines:
1. Translate the subtitles **faithfully** from English to Malayalam **without losing meaning or nuance**.
2. The Malayalam output should **sound natural, engaging, and conversational**, as if it were **spoken dialogue in a Malayalam-dubbed movie**.
3. **Preserve emotional tone, humor, sarcasm, and intensity** from the original.
4. Adapt **cultural references or idioms** to Malayalam context where appropriate.
5. Keep **sentence length and pacing** similar to match lip-sync and scene flow.
6. Do **not** provide explanations or transliterations — only the translated lines.
7. Maintain the **exact line numbering** and **line breaks**.

---

### Output Format:
Line [number]:
[Malayalam translation]

Line [number]:
[Malayalam translation]

(Ensure each line is separated exactly as in the input.)
"""

    
    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # Request payload
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0]:
                    if 'parts' in result['candidates'][0]['content']:
                        translated_text = result['candidates'][0]['content']['parts'][0]['text']
                        return translated_text
            
            return "Error: Could not parse response from Gemini API"
        else:
            error_message = response.json().get('error', {}).get('message', 'Unknown error')
            return f"Error: {response.status_code} - {error_message}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Main interface
if uploaded_file:
    # Read and parse SRT file
    content = uploaded_file.read().decode('utf-8')
    movie_name = Path(uploaded_file.name).stem
    
    subtitles = parse_srt(content)
    
    if subtitles:
        st.success(f"✅ Loaded {len(subtitles)} subtitles from '{movie_name}'")
        
        # Create two columns for line selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_line = st.number_input(
                "Start Line Number", 
                min_value=1, 
                max_value=len(subtitles),
                value=1,
                step=1
            )
        
        with col2:
            end_line = st.number_input(
                "End Line Number (optional)", 
                min_value=0, 
                max_value=len(subtitles),
                value=0,
                step=1,
                help="Leave as 0 to select only start line"
            )
        
        # Context range slider
        # st.subheader("Context Settings")
        # context_range = st.slider(
        #     "Include surrounding lines for context (before and after)",
        #     min_value=0,
        #     max_value=10,
        #     value=3,
        #     help="This helps Gemini understand the scene better"
        # )
        context_range = 5
        
        # Display selected subtitles
        if st.button("🔍 Preview Selection", type="primary"):
            actual_end = end_line if end_line > 0 else start_line
            selected, formatted = format_selected_subtitles(subtitles, start_line, actual_end)
            
            if selected:
                st.subheader("Selected Subtitles:")
                st.code(formatted, language=None)
                
                # Store in session state for translation
                st.session_state['selected'] = selected
                st.session_state['movie_name'] = movie_name
                st.session_state['start'] = start_line
                st.session_state['end'] = actual_end
                st.session_state['context_range'] = context_range
            else:
                st.error("No subtitles found in the selected range")
        
        # Translation button
        if 'selected' in st.session_state and api_key:
            if st.button("🌐 Translate to Malayalam", type="secondary"):
                with st.spinner("Translating with Gemini AI..."):
                    # Get context subtitles
                    context_start = max(1, st.session_state['start'] - st.session_state['context_range'])
                    context_end = min(len(subtitles), st.session_state['end'] + st.session_state['context_range'])
                    
                    context_subs = [s for s in subtitles if context_start <= s['number'] <= context_end]
                    
                    # Translate
                    translation = translate_with_gemini(
                        st.session_state['selected'],
                        st.session_state['movie_name'],
                        context_subs,
                        api_key
                    )
                    
                    st.subheader("Malayalam Translation:")
                    st.markdown(translation)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Translation",
                        data=translation,
                        file_name=f"{st.session_state['movie_name']}_ml_{st.session_state['start']}-{st.session_state['end']}.txt",
                        mime="text/plain"
                    )
        elif 'selected' in st.session_state and not api_key:
            st.warning("⚠️ Please enter your Gemini API Key to translate")
    else:
        st.error("Could not parse SRT file. Please check the format.")
# else:
#     st.info("👆 Please upload an SRT file to get started")
    
    

import os
import time
import logging
import streamlit as st
from typing import Dict, List
import google.generativeai as genai
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SrtTranslator:
    def __init__(self, api_key: str, source_language: str = "Turkish", target_language: str = "Malayalam", batch_size: int = 60):
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.batch_size = batch_size
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-lite",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,  # Increased for batch processing
                }
            )
            
            self.translation_prompt = f"""Translate the following {source_language} subtitle lines from a movie to proper {target_language} script. 
Maintain the exact same number of lines in your translation as in the original text.

Guidelines:
1. IMPORTANT: You MUST use PROPER {target_language} UNICODE script characters only (NOT romanized/transliterated text)
2. Use authentic, colloquial {target_language} expressions rather than literal translations
3. Preserve character names, places, and technical terms in their original form
4. Match the tone, emotion, and register (formal/informal) of the original dialogue
5. Keep translations concise for proper subtitle display
6. Adapt cultural references appropriately for a {target_language}-speaking audience
7. Maintain the original meaning and intent of the dialogue

Each line is separated by a line break. Translate each line and keep the same number of lines in your response.

{source_language} subtitles:
{{text}}

Provide only the {target_language} translations in proper {target_language} Unicode script, with each line separated by a line break, matching the original number of lines exactly.
"""
        except Exception as e:
            st.error(f"Failed to initialize Gemini API: {str(e)}")
            raise

    def translate_batch(self, text_lines: List[str], progress_bar=None) -> List[str]:
        """Translate a batch of text lines while preserving the line structure."""
        if not text_lines:
            return []
            
        # Join lines with newlines for batch translation
        batch_text = "\n".join(text_lines)
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    self.translation_prompt.format(text=batch_text)
                )
                translated_text = response.text.strip()
                
                # Split the translated text back into lines
                translated_lines = translated_text.split('\n')
                
                # Verify we have the same number of lines
                if len(translated_lines) != len(text_lines):
                    logging.warning(f"Translation returned {len(translated_lines)} lines, expected {len(text_lines)}. Adjusting...")
                    # Adjust by truncating or padding
                    if len(translated_lines) > len(text_lines):
                        translated_lines = translated_lines[:len(text_lines)]
                    else:
                        translated_lines.extend([""] * (len(text_lines) - len(translated_lines)))
                
                return translated_lines
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logging.warning(f"Batch translation attempt {attempt + 1} failed. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"Batch translation failed after {self.max_retries} attempts: {str(e)}")
                    # Return empty translations on failure
                    return ["TRANSLATION_ERROR"] * len(text_lines)
    
    def parse_srt(self, srt_content: str) -> List[Dict]:
        """Parse SRT content into structured format."""
        try:
            blocks = srt_content.strip().split('\n\n')
            parsed_blocks = []
            
            for block in blocks:
                lines = block.split('\n')
                if len(lines) >= 3:  # Valid subtitle blocks have at least 3 lines
                    try:
                        subtitle_id = int(lines[0])
                        timestamp = lines[1]
                        text = '\n'.join(lines[2:])
                        
                        parsed_blocks.append({
                            'id': subtitle_id,
                            'timestamp': timestamp,
                            'text': text
                        })
                    except ValueError:
                        logging.warning(f"Skipping invalid block: {block}")
                        
            logging.info(f"Parsed {len(parsed_blocks)} subtitle blocks")
            return parsed_blocks
            
        except Exception as e:
            logging.error(f"Error parsing SRT content: {str(e)}")
            raise
    
    def translate_srt_content(self, srt_content: str, progress_callback=None) -> str:
        """Translate SRT content in batches and return the translated content."""
        try:
            # Parse input SRT
            subtitles = self.parse_srt(srt_content)
            
            # Process subtitles in batches
            total_blocks = len(subtitles)
            translated_content = ""
            
            # Create batches
            for i in range(0, total_blocks, self.batch_size):
                batch_end = min(i + self.batch_size, total_blocks)
                current_batch = subtitles[i:batch_end]
                
                logging.info(f"Translating batch {i//self.batch_size + 1}, subtitles {i+1}-{batch_end} of {total_blocks}")
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i / total_blocks)
                
                # Extract text from each subtitle in the batch
                batch_texts = [subtitle['text'] for subtitle in current_batch]
                
                # Translate the batch
                translated_texts = self.translate_batch(batch_texts)
                
                # Update subtitles with translations
                for j, translated_text in enumerate(translated_texts):
                    subtitles[i + j]['translated_text'] = translated_text
                
                # Add a small delay between batches to avoid rate limiting
                if batch_end < total_blocks:
                    time.sleep(1)
            
            # Format translated SRT
            for subtitle in subtitles:
                translated_content += f"{subtitle['id']}\n"
                translated_content += f"{subtitle['timestamp']}\n"
                translated_content += f"{subtitle['translated_text']}\n\n"
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0)
                
            return translated_content
            
        except Exception as e:
            logging.error(f"Translation process failed: {str(e)}")
            raise

# Streamlit UI
def main():
    st.set_page_config(
        page_title="SRT Translator",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 SRT Subtitle Translator")
    st.markdown("Upload an SRT file to translate subtitles using Google's Gemini AI")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input with password masking
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    
    # Source language selection
    source_language = st.sidebar.selectbox(
        "Source Language",
        options=["Turkish"],
        index=0
    )
    
    # Target language selection
    target_language = st.sidebar.selectbox(
        "Target Language",
        options=["Malayalam"],
        index=0
    )
    
    # Batch size slider
    batch_size = st.sidebar.slider(
        "Batch Size (subtitles per API call)",
        min_value=10,
        max_value=100,
        value=60,
        step=10
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload SRT File", type=["srt"])
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
        
        # Read file content
        srt_content = uploaded_file.getvalue().decode("utf-8")
        
        # Show sample of content
        st.subheader("Sample of SRT Content")
        st.text(srt_content[:500] + "..." if len(srt_content) > 500 else srt_content)
        
        # Translation button
        if st.button("Translate Subtitles"):
            if not api_key:
                st.error("Please enter a valid Gemini API Key")
            else:
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Translation progress: {progress*100:.1f}%")
                    
                    status_text.text("Initializing translator...")
                    
                    # Initialize translator
                    translator = SrtTranslator(
                        api_key=api_key,
                        source_language=source_language,
                        target_language=target_language,
                        batch_size=batch_size
                    )
                    
                    status_text.text("Starting translation...")
                    
                    # Translate content
                    translated_content = translator.translate_srt_content(
                        srt_content,
                        progress_callback=update_progress
                    )
                    
                    # Show success message
                    st.success("Translation completed!")
                    
                    # Display sample of translated content
                    st.subheader("Sample of Translated SRT")
                    st.text(translated_content[:500] + "..." if len(translated_content) > 500 else translated_content)
                    
                    # Provide download button
                    st.download_button(
                        label="Download Translated SRT",
                        data=translated_content,
                        file_name=f"translated_{uploaded_file.name}",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
    
    # Add information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This app translates SRT subtitle files using Google's Gemini AI.
        
        **Features:**
        - Batch processing for efficiency
        - Preserves SRT formatting
        - Maintains subtitle timing
        - Natural, conversational translations
        
        **Note:** You need a valid Gemini API key to use this app.
        """
    )

if __name__ == "__main__":
    main()

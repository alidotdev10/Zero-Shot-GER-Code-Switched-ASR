import gradio as gr
from groq import Groq
import whisper
import warnings
import os
import soundfile as sf
from datasets import load_dataset

# Suppress minor warnings to keep the terminal clean
warnings.filterwarnings("ignore")

# --- 1. DATASET DOWNLOADER ---
print("Checking Hugging Face dataset...")
os.makedirs("dataset_audio", exist_ok=True)
ds = load_dataset("MohamedRashad/multilingual-tts", split="train")

saved_audio_paths = []
# Extracting a small batch for the UI
for i in range(3):
    audio_data = ds[i]["audio"]
    # We use a simple relative path to bypass Gradio's security blocks
    file_path = f"dataset_audio/sample_{i}.wav"
    
    # Only write the file if it doesn't already exist to save time on restarts
    if not os.path.exists(file_path):
        sf.write(file_path, audio_data["array"], audio_data["sampling_rate"])
    
    # Gradio requires a list of lists for examples
    saved_audio_paths.append([file_path])

print("Success! Dataset audio is ready.")
# ------------------------------

# 2. Initialize the free Groq Client
# PLEASE REVOKE THIS KEY TOMORROW!
API_KEY = "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" 
client = Groq(api_key=API_KEY)

# 3. Initialize Whisper (Using 'base' so it runs quickly on your CPU)
print("Loading Whisper ASR model...")
asr_model = whisper.load_model("base")

def process_audio(audio_path):
    if not audio_path:
        return "Error: No audio provided.", ""

    print(f"Processing audio at: {audio_path}")

    # Step 1: Transcribe using Groq's Whisper Large V3 (State-of-the-art for Code-Switching)
    print("Transcribing audio via Groq Whisper-Large-V3...")
    try:
        with open(audio_path, "rb") as audio_file:
            asr_result = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0  # Keep it deterministic
            )
        raw_transcript = asr_result.text
    except Exception as e:
        return f"ASR Error: {str(e)}", ""

    # Step 2: Code-Switched Generative Error Correction via LLM
    print("Running LLM Error Correction for Code-Switching...")
    prompt = (
"You are an expert NLP researcher specializing in highly multilingual, code-switched ASR error correction. "
        "The following text is a raw machine transcription from a dataset containing 28 different languages. "
        "The speaker frequently switches between multiple languages mid-sentence (code-switching). "
        "Because of 'Language Dominance Bias', the ASR model often panics and forces words from one language into the script or phonetics of another dominant language.\n\n"
        "Your task:\n"
        "1. Dynamically analyze the raw transcript to deduce which languages the speaker was actually attempting to use.\n"
        "2. Identify cross-language phonetic corruptions (where words from Language A are incorrectly written using characters or sounds of Language B).\n"
        "3. Restore the true, intended multi-lingual words in their correct respective scripts or standard romanized representations.\n"
        "4. CRITICAL: Absolute prohibition against total translation. Under no circumstances should you translate the entire sentence into English or any single dominant language. Preserve the multi-lingual code-switching perfectly.\n\n"
        "Output ONLY the final corrected mixed-language sentence. Do not include any explanation or metadata.\n\n"
        f"Raw Transcript: {raw_transcript}"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2, 
        )
        corrected_transcript = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        corrected_transcript = f"LLM Error: {str(e)}"

    return raw_transcript, corrected_transcript


# 4. Build the UI
with gr.Blocks(title="Generative Error Correction for ASR") as demo:
    gr.Markdown("# Generative Error Correction for ASR via LLM")
    gr.Markdown(
        "**Context:** Testing zero-shot error correction on noisy acoustic streams. "
        "Click an example below or upload your own audio to compare raw Whisper output vs. LLM-corrected output."
    )

    with gr.Row():
        # Notice we are back to type="filepath" because it handles both uploads and examples cleanly
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath")
    
    # The Examples block is now properly fed the relative paths
    gr.Examples(
        examples=saved_audio_paths,
        inputs=audio_input,
        label="Click a downloaded dataset sample below to test:"
    )
    
    with gr.Row():
        raw_output = gr.Textbox(label="Baseline (Whisper-Base) Output", lines=4)
        corrected_output = gr.Textbox(label="Corrected (LLaMA-3) Output", lines=4)

    submit_btn = gr.Button("Run Error Correction Pipeline", variant="primary")
    
    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[raw_output, corrected_output],
    )

if __name__ == "__main__":
    demo.launch()
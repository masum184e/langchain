# Multimodality

Multimodal systems process more than one type of data. In the context of LangChain, this typically means combining:

- **Text ↔ Image:** Describe images, generate images from text.
- **Text ↔ Audio:** Convert audio to text (speech recognition), or text to speech.
- **Text ↔ Video:** Summarize or query video content.
- **Text ↔ Files/Documents:** Extract, summarize, or ask questions about file content (e.g., PDF, DOCX).

LangChain leverages tool integrations, custom agents, and multi-input chains to enable this.

Multimodality in LangChain refers to the ability to handle and integrate multiple types of input and output data—text, images, audio, video, etc.—in language model applications. LangChain provides tooling to connect large language models (LLMs) with non-text modalities, enabling richer, more interactive, and more versatile AI applications.

## How LangChain Supports Multimodality

### 1. Tool Integration

You can use tools that handle image/audio processing and let the LLM orchestrate them:

- `image_captioning_tool`
- `image_generation_tool`
- `speech_to_text_tool`

LangChain Agents can call tools as needed, switching between modalities.

### 2. Custom Agents and Tools

You can define custom tools that handle images, video, or audio. LangChain's agent framework lets LLMs decide when and how to use these tools.

### 3. Output Parsers & Input Loaders

LangChain supports parsing and loading of diverse formats via:

- `UnstructuredLoader` for PDFs, Word files, HTML.
- `ImageLoader`, `AudioLoader`, etc.

## Example

### Image Description

**Scenario:** an agent that receives an image and describes its content using a captioning tool.

```py
@tool
def image_captioning_tool(image_path: str) -> str:
    """Generates a caption for the given image file path."""
    # Placeholder logic (replace with real model like BLIP or CLIP)
    return "A group of people walking in a park during sunset."

tools = [
    Tool(
        name="ImageCaptioningTool",
        func=image_captioning_tool,
        description="Use this tool to describe the content of an image."
    )
]

agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

image_path = "sample_image.jpg"
response = agent.run(f"What is shown in the image located at {image_path}?")
```

### Video Summarization

```py
def extract_audio(video_path: str, audio_output: str):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output)

extract_audio("lecture.mp4", "lecture_audio.wav")

model = whisper.load_model("base")
transcription = model.transcribe("lecture_audio.wav")["text"]

prompt = f"Summarize the following lecture:\n\n{transcription}"
response = llm.predict(prompt)

print(response)
```

### More Realistic Scenarios

| Modality     | Example Use Case                               | Tool Integration                       |
| ------------ | ---------------------------------------------- | -------------------------------------- |
| Text + Image | Captioning, object recognition, scene analysis | OpenAI Vision, BLIP, CLIP              |
| Text + Audio | Voice command processing, call transcription   | Whisper, Google Speech-to-Text         |
| Text + Files | QA on PDFs or scanned documents                | Unstructured.io, PyMuPDF, OCR          |
| Text + Video | Summarize or search inside YouTube videos      | Whisper + Frame Extraction + LangChain |

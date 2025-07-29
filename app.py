# server.py
from audio_processor import AudioProcessor
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
from utils import process_frame, SAMPLE_RATE, FRAME_SIZE

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Load model (Hugging Face Whisper large-v3) ---
model_id = "openai/whisper-large-v3"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"

print("Loading Whisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs={"language": "en"}
)

print("Model loaded!")

FRAMES_PER_BUFFER = 10
VAD_SPEECH_THRESHOLD = 0.6
SILENCE_TIMEOUT_SECONDS = 1.5
MAX_RECORDING_SECONDS = 30
CHUNK_SIZE_BYTES = FRAME_SIZE * 2 * FRAMES_PER_BUFFER
SILENCE_FRAMES_THRESHOLD = int((SILENCE_TIMEOUT_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
MAX_RECORDING_FRAMES = int((MAX_RECORDING_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
BYTES_PER_SAMPLE = 2

@app.get("/")
async def demo_interface():
    return FileResponse("index.html")

audio_processor = AudioProcessor(pipe=asr_pipe)

# --- WebSocket endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    audio_buffer = bytearray()
    recording_buffer = []
    is_recording = False
    silence_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            if len(audio_buffer) < CHUNK_SIZE_BYTES:
                continue

            chunk_bytes = audio_buffer[:CHUNK_SIZE_BYTES]
            del audio_buffer[:CHUNK_SIZE_BYTES]

            frames = []
            speech_frame_count = 0
            frame_byte_size = FRAME_SIZE * BYTES_PER_SAMPLE
            for i in range(0, len(chunk_bytes), frame_byte_size):
                frame_data = chunk_bytes[i:i + frame_byte_size]
                if len(frame_data) < frame_byte_size:
                    continue # Bỏ qua frame cuối nếu không đủ byte

                # Gọi hàm xử lý mới, thuần túy
                is_speech, waveform = process_frame(frame_data)
                
                if waveform is not None:
                    frames.append(waveform)
                    if is_speech:
                        speech_frame_count += 1
            
            if not frames:
                continue

            speech_probability = speech_frame_count / len(frames)

            if is_recording:
                recording_buffer.extend(frames)
                if speech_probability > VAD_SPEECH_THRESHOLD:
                    silence_counter = 0
                else:
                    silence_counter += len(frames)

                if silence_counter > SILENCE_FRAMES_THRESHOLD or len(recording_buffer) > MAX_RECORDING_FRAMES:
                    print(f"[*] Dừng ghi âm. Tổng số khung: {len(recording_buffer)}")
                    final_waveform = torch.cat(recording_buffer, dim=1)
                    int16_tensor = (final_waveform.squeeze() * 32767).to(torch.int16)
                    audio_bytes = int16_tensor.cpu().numpy().tobytes()
                    transcript = await audio_processor.process_audio(audio_bytes)
                    await websocket.send_text(transcript)

                    is_recording = False
                    recording_buffer.clear()
                    silence_counter = 0
            
            elif speech_probability > VAD_SPEECH_THRESHOLD:
                print("[*] Bắt đầu ghi âm...")
                is_recording = True
                recording_buffer.extend(frames)
                silence_counter = 0
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed.")

if __name__ == "__main__":
    import socket
    import ssl

    import uvicorn

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="ssl/cert.pem", keyfile="ssl/key.pem")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        ssl_keyfile="ssl/key.pem",
        ssl_certfile="ssl/cert.pem",
    )

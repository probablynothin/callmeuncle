import asyncio
import os
import traceback

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from functions.storeAddress import (
    add_complaint,
    add_complaint_tool,
    check_for_complaint,
    check_for_complaint_tool,
    get_complaint_details,
    get_complaint_details_tool,
)

# Load environment variables
load_dotenv()

SEND_SAMPLE_RATE = 16000  # Sample rate for sending mic input
RECEIVE_SAMPLE_RATE = 24000  # Sample rate for audio output
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

# Initialize the GenAI client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1alpha"},
)

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[
        types.Tool(
            function_declarations=[
                add_complaint_tool,
                check_for_complaint_tool,
                get_complaint_details_tool,
            ]
        )
    ],
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="You are a Customer Representative named Om"
                ", Check for complaints and help store the names and addresses of customers, "
                "who want to register a complaint, and help check if a complaint already exists."
            )
        ]
    ),
)


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None  # Queue for audio we receive from Gemini
        self.audio_out_queue = None  # Queue for mic audio we send to Gemini
        self.session: AsyncSession | None = None
        # A buffer to store leftover audio from bigger-than-requested chunks.
        self.play_buffer = bytearray()

    async def send_text(self):
        """Send user-entered text to the model."""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(text or ".", end_of_turn=True)

    async def send_realtime(self):
        """Continuously send mic audio chunks to the model."""
        while True:
            msg = await self.audio_out_queue.get()
            await self.session.send(msg)

    async def listen_audio(self):
        """Capture microphone input and enqueue it to be sent to Gemini."""

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio input error: {status}")
            try:
                self.audio_out_queue.put_nowait(
                    {"data": indata.copy().tobytes(), "mime_type": "audio/pcm"}
                )
            except asyncio.QueueFull:
                print("⚠️ Queue is full. Dropping mic audio chunk.")

        print("🎤 Listening... Press Ctrl+C to exit.")
        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                while True:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("🎤 Stopped listening to mic input.")

    async def receive_chunks(self):
        """Receive chunks from Gemini and put them into the playback queue."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    await self.audio_in_queue.put(response.data)
                elif response.text:
                    print(response.text, end="", flush=True)
                elif response.tool_call:
                    print("tool call received", response.tool_call)

                    function_responses = []
                    if response.tool_call.function_calls:
                        for function_call in response.tool_call.function_calls:
                            name = function_call.name
                            args = function_call.args
                            # Extract the numeric part from Gemini's function call ID
                            call_id = function_call.id
                            if name == "check_for_complaint":
                                if not args:
                                    print("Missing required parameter 'name'")
                                    continue
                                argname = args["name"]
                                boolean = check_for_complaint(argname)
                                function_responses.append(
                                    {
                                        "name": "check_for_complaint",
                                        "response": {"exists": boolean},
                                        "id": call_id,
                                    }
                                )
                            elif name == "add_complaint":
                                if not args:
                                    print(
                                        "Missing required parameters 'name' and 'address'"
                                    )
                                    continue
                                argname = args["name"]
                                argaddress = args["address"]

                                add_complaint(argname, argaddress)
                                response = (
                                    f"Stored the address of {argname} as {argaddress}"
                                )
                                function_responses.append(
                                    {
                                        "name": "add_complaint",
                                        "response": {"response": response},
                                        "id": call_id,
                                    }
                                )
                            elif name == "get_complaint_details":
                                if not args:
                                    print("Missing required parameter 'name'")
                                    continue
                                argname = args["name"]
                                address = get_complaint_details(argname)

                                function_responses.append(
                                    {
                                        "name": "get_complaint_details",
                                        "response": {
                                            "name": argname,
                                            "address": address,
                                        },
                                        "id": call_id,
                                    }
                                )
                            else:
                                print(f"Unknown function name: {function_call.name}")

                    await self.session.send(function_responses)

    def fill_outdata(self, outdata: np.ndarray):
        """
        Fill 'outdata' from our ring buffer.
        If we don't have enough audio in 'play_buffer', read from audio_in_queue.
        """
        bytes_needed = outdata.nbytes  # e.g., frames * channels * 2 bytes
        # If we don't have enough leftover audio in 'play_buffer', pull more from the queue.
        while len(self.play_buffer) < bytes_needed:
            try:
                # This call is non-blocking: if empty, it raises QueueEmpty
                new_chunk = self.audio_in_queue.get_nowait()
                self.play_buffer += new_chunk
            except asyncio.QueueEmpty:
                # No more audio available right now; fill the remainder with zeros.
                missing = bytes_needed - len(self.play_buffer)
                self.play_buffer += b"\x00" * missing
                break

        # Now we have at least 'bytes_needed' in the buffer (or we've zero-filled).
        chunk = self.play_buffer[:bytes_needed]
        self.play_buffer = self.play_buffer[bytes_needed:]

        # Convert chunk to int16 array and reshape to match (frames, channels).
        outdata[:] = np.frombuffer(chunk, dtype="int16").reshape(-1, 1)

    async def play_audio(self):
        """Play audio data from the ring buffer in the OutputStream callback."""

        def callback(outdata, frames, time, status):
            if status:
                print(f"Audio output error: {status}")
            self.fill_outdata(outdata)

        try:
            with sd.OutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                while True:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("🎧 Stopped audio playback.")

    async def run(self):
        """Main loop to manage tasks and handle graceful exit."""
        try:
            async with (
                client.aio.live.connect(
                    model=MODEL,
                    config=config,
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Async queues
                self.audio_in_queue = asyncio.Queue()
                self.audio_out_queue = asyncio.Queue(maxsize=5)

                # Start tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_chunks())
                tg.create_task(self.play_audio())

                await send_text_task  # Wait for text input or 'q'
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            print("\nExiting...")
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        # response_modality = (
        #     input("Enter response modality (a for audio, t for text): ").strip().lower()
        # )
        # if response_modality == "a":
        #     config["response_modalities"] = ["AUDIO"]
        # elif response_modality == "t":
        #     config["response_modalities"] = ["TEXT"]
        # else:
        #     print("Invalid input. Defaulting to Audio response.")
        #     config["response_modalities"] = ["AUDIO"]

        main = AudioLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting... Goodbye!")

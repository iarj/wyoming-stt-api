import asyncio
import logging
import wave
from io import BytesIO

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_stt_api.clients.openai import OpenAIClient

logger = logging.getLogger(__name__)


class WyomingEventHandler(AsyncEventHandler):
    def __init__(
        self,
        openai_client: OpenAIClient,
        max_audio_duration_s: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        super().__init__(reader, writer)
        self._openai_client = openai_client
        self._max_audio_duration_s = max_audio_duration_s
        self._wave_file: wave.Wave_write | None = None
        self._buffer: BytesIO = BytesIO()

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            self._start_audio(AudioStart.from_event(event))
            return True

        if AudioChunk.is_type(event.type):
            self._add_audio_chunk(AudioChunk.from_event(event))
            return True

        if AudioStop.is_type(event.type):
            # get the duration before file is closed
            duration = self._audio_duration
            self._stop_audio()
            if duration <= self._max_audio_duration_s:
                await self._transcribe_buffer()
            else:
                logger.error(
                    f"Audio exceeds max duration of {self._max_audio_duration_s}s and will not be transcribed"
                )
            return False

        if Transcribe.is_type(event.type):
            # We transcribe in Romanian only (forced in Settings/OpenAIClient).
            return True

        if Describe.is_type(event.type):
            await self._send_info()
            return True

        return False

    def _start_audio(self, start: AudioStart):
        self._buffer = BytesIO()
        self._wave_file = wave.open(self._buffer, "wb")
        self._wave_file.setframerate(start.rate)
        self._wave_file.setsampwidth(start.width)
        self._wave_file.setnchannels(start.channels)

    def _add_audio_chunk(self, chunk: AudioChunk):
        if self._wave_file is None:
            raise ValueError("Audio not started")

        self._wave_file.writeframes(chunk.audio)

    def _stop_audio(self):
        if self._wave_file is None:
            raise ValueError("Audio not started")

        logger.info(f"Received audio with duration {self._audio_duration:.2f}s")
        self._wave_file.close()
        self._wave_file = None

    async def _transcribe_buffer(self):
        text = self._openai_client.speech_to_text(self._buffer, file_extension="wav")
        await self.write_event(Transcript(text=text).event())

    async def _send_info(self):
        info = Info(
            asr=[
                AsrProgram(
                    name="Speech-to-text API",
                    description=None,
                    version=None,
                    attribution=Attribution(
                        name="wyoming-stt-api",
                        url="https://github.com/gabrielwong159/wyoming-stt-api",
                    ),
                    installed=True,
                    models=[
                        AsrModel(
                            name=self._openai_client.model_name,
                            description=None,
                            version=None,
                            attribution=Attribution(
                                name="OpenAI",
                                url="https://platform.openai.com/docs/models",
                            ),
                            installed=True,
                            languages=["ro"],
                        )
                    ],
                )
            ],
        )
        await self.write_event(info.event())

    @property
    def _audio_duration(self) -> float:
        if self._wave_file is None:
            raise ValueError("Audio not started")

        return self._wave_file.getnframes() / self._wave_file.getframerate()

import asyncio
import logging

from wyoming.server import AsyncTcpServer

from wyoming_stt_api.clients.openai import OpenAIClient
from wyoming_stt_api.services.wyoming import WyomingEventHandler
from wyoming_stt_api.settings import Settings

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s %(asctime)s %(message)s"
)

settings = Settings()
openai_client = OpenAIClient(
    api_key=settings.openai_api_key, model=settings.openai_model, language=settings.openai_language,  # <-- ADD
)
server = AsyncTcpServer(settings.server_host, settings.server_port)


def create_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    return WyomingEventHandler(
        openai_client, settings.max_audio_duration_s, reader, writer
    )

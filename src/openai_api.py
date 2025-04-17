#!/usr/bin/env python
#
# Copyright (C) 2024 SIP Point Consulting SRL
#
# This file is part of the OpenSIPS AI Voice Connector project
# (see https://github.com/OpenSIPS/opensips-ai-voice-connector-ce).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
OpenAI WS communication
"""

import json
import base64
import logging
import asyncio
from queue import Empty
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from ai import AIEngine
from codec import get_codecs, CODECS, UnsupportedCodec
from config import Config

OPENAI_API_MODEL = "gpt-4o-realtime-preview-2024-12-17"
OPENAI_URL_FORMAT = "wss://api.openai.com/v1/realtime?model={}"


class OpenAI(AIEngine):  # pylint: disable=too-many-instance-attributes

    logging.info(" OPENAI_API -> Implements WS communication with OpenAI ")

    def __init__(self, call, cfg):
        self.codec = self.choose_codec(call.sdp)
        self.queue = call.rtp
        self.call = call
        self.ws = None
        self.session = None
        self.intro = None
        self.transfer_to = None
        self.transfer_by = None
        self.cfg = Config.get("openai", cfg)
        self.model = self.cfg.get("model", "OPENAI_API_MODEL",
                                  OPENAI_API_MODEL)
        self.url = self.cfg.get("url", "OPENAI_URL",
                                OPENAI_URL_FORMAT.format(self.model))
        self.key = self.cfg.get(["key", "openai_key"], "OPENAI_API_KEY")
        self.voice = self.cfg.get(["voice", "openai_voice"],
                                  "OPENAI_VOICE", "alloy")
        self.instructions = self.cfg.get("instructions", "OPENAI_INSTRUCTIONS")
        self.intro = self.cfg.get("welcome_message", "OPENAI_WELCOME_MSG")
        self.transfer_to = self.cfg.get("transfer_to", "OPENAI_TRANSFER_TO")
        self.transfer_by = self.cfg.get("transfer_by", "OPENAI_TRANSFER_BY", self.call.to)

        # normalize codec
        if self.codec.name == "mulaw":
            self.codec_name = "g711_ulaw"
        elif self.codec.name == "alaw":
            self.codec_name = "g711_alaw"

    def choose_codec(self, sdp):
        logging.info(" OPENAI_API -> Returns the preferred codec from a list ")
        codecs = get_codecs(sdp)
        priority = ["pcma", "pcmu"]
        cmap = {c.name.lower(): c for c in codecs}
        for codec in priority:
            if codec in cmap:
                return CODECS[codec](cmap[codec])

        raise UnsupportedCodec("No supported codec found")

    def get_audio_format(self):
        logging.info(" OPENAI_API -> Returns the corresponding audio format ")
        return self.codec_name
    
    # async def start(self):
    #     headers = {
    #         "Authorization": f"Bearer {self.key}",
    #         "OpenAI-Beta": "realtime=v1"
    #     }
    #     print("Tentando conectar")
    #     self.ws = await connect(self.url, additional_headers=headers)
    #     print("Conexão feita com sucesso")

    
    async def start(self):
        logging.info(" OPENAI_API -> Starts OpenAI connection and logs messages ")
        openai_headers = {
                "Authorization": f"Bearer {self.key}",
                "OpenAI-Beta": "speech-to-speech"
        }
        logging.info(" OPENAI_API -> conectando ao websocket ")
        
        headers = {
            "Authorization": f"Bearer {self.key}",
            "OpenAI-Beta": "realtime=v1"
        }
        logging.info("URL ==> " + self.url)
        logging.info("HEADERS ==> " + json.dumps(headers))
        self.ws = await connect(self.url, additional_headers=headers)
        # self.ws = websocket.WebSocketApp(
        #     self.url,
        #     header=openai_headers,
        #     on_open=logging.info("OPENAI_API -> conectado a openai."),
        #     on_message=self.on_message,
        #     on_error=self.on_error,
        #     on_close=self.on_close
        # )
        logging.info(" OPENAI_API -> conectado ")
        try:
            json_result = json.loads(await self.ws.recv())
        except ConnectionClosedOK:
            logging.info(" OPENAI_API ->WS Connection with OpenAI is closed")
            return
        except ConnectionClosedError as e:
            logging.error(e)
            return

        self.session = {
            "turn_detection": {
                "type": self.cfg.get("turn_detection_type",
                                     "OPENAI_TURN_DETECT_TYPE",
                                     "server_vad"),
                "silence_duration_ms": int(self.cfg.get(
                    "turn_detection_silence_ms",
                    "OPENAI_TURN_DETECT_SILENCE_MS",
                    200)),
                "threshold": float(self.cfg.get(
                    "turn_detection_threshold",
                    "OPENAI_TURN_DETECT_THRESHOLD",
                    0.5)),
                "prefix_padding_ms": int(self.cfg.get(
                    "turn_detection_prefix_ms",
                    "OPENAI_TURN_DETECT_PREFIX_MS",
                    200)),
            },
            "input_audio_format": self.get_audio_format(),
            "output_audio_format": self.get_audio_format(),
            "input_audio_transcription": {
                "model": "whisper-1",
            },
            "voice": self.voice,
            "temperature": float(self.cfg.get("temperature",
                                              "OPENAI_TEMPERATURE", 0.8)),
            "max_response_output_tokens": self.cfg.get("max_tokens",
                                                       "OPENAI_MAX_TOKENS",
                                                       "inf"),
            "tools": [
                {
                    "type": "function",
                    "name": "terminate_call",
                    "description":
                        "Call me when any of the session's parties want "
                        "to terminate the call."
                        "Always say goodbye before hanging up."
                        "Send the audio first, then call this function.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
                {
                    "type": "function",
                    "name": "transfer_call",
                    "description": "call the function if a request was received to transfer a call with an operator, a person",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
            ],
            "tool_choice": "auto",
        }
        if self.instructions:
            self.session["instructions"] = self.instructions

        try:
            logging.info("OPENAI_API -> Enviando session update de início de conversação para OPENAI")
            json_to_send = json.dumps({"type": "session.update", "session": self.session})
            logging.info("json to send => " + json_to_send)
            await self.ws.send(json_to_send)
            if self.intro:
                self.intro = {
                    "instructions": "Please greet the user with the following: " +
                    self.intro
                }
                logging.info("OPENAI_API -> Entrou no self.intro")
                self.ws.send(json.dumps({"type": "response.create", "response": self.intro}))
            
            logging.info("OPENAI_API -> enviado")
            await self.handle_command()
        except ConnectionClosedError as e:
            logging.error(f"Error while communicating with OpenAI: {e}. Terminating call.")
            self.terminate_call()
        except Exception as e:
            logging.error(f"Unexpected error during session: {e}. Terminating call.")
            self.terminate_call()

    async def handle_command(self):  # pylint: disable=too-many-branches
        logging.info(" OPENAI_API -> Handles a command from the server ")
        leftovers = b''
        t = ""
        while t != "response.done":
            logging.info(" OPENAI_API -> loop para checagem de mensagens da OPENAI ")
            smsg = await self.ws.recv()
            msg = json.loads(smsg)
            logging.info("OPENAI_API -> " + smsg)
            t = msg["type"]
            if t == "response.audio.delta":
                logging.info("OPENAI_API -> response.audio.delta")
                logging.info("OPENAI_API -> resposta delta => " + msg["delta"])
                media = base64.b64decode(msg["delta"])
                packets, leftovers = await self.run_in_thread(
                    self.codec.parse, media, leftovers)
                for packet in packets:
                    logging.info("OPENAI_API -> Enfileirando os dados")
                    self.queue.put_nowait(packet)
            elif t == "response.audio.done":
                logging.info("OPENAI_API -> response.audio.done")
                logging.info(t)
                if len(leftovers) > 0:
                    packet = await self.run_in_thread(
                            self.codec.parse, None, leftovers)
                    self.queue.put_nowait(packet)
                    leftovers = b''

            elif t == "conversation.item.created":
                logging.info("OPENAI_API -> conversation.item.created")
                if msg["item"].get('status') == "completed":
                    self.drain_queue()
            elif t == "conversation.item.input_audio_transcription.completed":
                logging.info("OPENAI_API -> conversation.item.input_audio_transcription.completed")
                logging.info(" OPENAI_API ->Speaker: %s", msg["transcript"].rstrip())
            elif t == "response.audio_transcript.done":
                logging.info("OPENAI_API -> response.audio_transcript.done")
                logging.info(" OPENAI_API ->Engine: %s", msg["transcript"])
            elif t == "response.function_call_arguments.done":
                logging.info("OPENAI_API -> response.function_call_arguments.done")
                if msg["name"] == "terminate_call":
                    logging.info(t)
                    self.terminate_call()
                elif msg["name"] == "transfer_call":
                    params = {
                        'key': self.call.b2b_key,
                        'method': "REFER",
                        'body': "",
                        'extra_headers': (
                            f"Refer-To: <{self.transfer_to}>\r\n"
                            f"Referred-By: {self.transfer_by}\r\n"
                        )
                    }
                    self.call.mi_conn.execute('ua_session_update', params)

            elif t == "error":
                logging.info("OPENAI_API -> error")
                logging.info(msg)
            else:
                logging.info(t)
                return
        
        logging.info(" OPENAI_API -> Passou do loop ")

    def on_message(self, message):
        logging.info("OPENAI_API -> message received")
        asyncio.run(self.handle_command(message))

    def on_error(self, error):
        logging.info("Erro:", error)

    def on_close(self, close_status_code, close_msg):
        logging.info("🔒 Conexão encerrada.", close_status_code, close_msg)

    def terminate_call(self):
        logging.info(" OPENAI_API -> Terminates the call ")
        self.call.terminated = True

    async def run_in_thread(self, func, *args):
        logging.info(" OPENAI_API -> Runs a function in a thread ")
        return await asyncio.to_thread(func, *args)

    def drain_queue(self):
        logging.info(" OPENAI_API -> Drains the playback queue ")
        count = 0
        try:
            while self.queue.get_nowait():
                count += 1
        except Empty:
            if count > 0:
                logging.info(" OPENAI_API ->dropping %d packets", count)

    async def send(self, audio):
        logging.info(" OPENAI_API -> Sends audio to OpenAI ")
        if not self.ws or self.call.terminated:
            return

        audio_data = base64.b64encode(audio)
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_data.decode("utf-8")
        }

        try:
            await self.ws.send(json.dumps(event))
        except ConnectionClosedError as e:
            logging.error(f"WebSocket connection closed: {e.code}, {e.reason}")
            self.terminate_call()
        except Exception as e:
            logging.error(f"Unexpected error while sending audio: {e}")
            self.terminate_call()

    async def close(self):
        await self.ws.close()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

# 1. Usar um atalho para gravar minha voz
# 2. Transcrever o audio para texto (em portugues) -> Whisper
# 3. De posse deste texto, quero jogar em uma LLM -> Agente
# 4. De possa da resposta da LLM, quero utilizar um modelo de TTS (API da OpenAI)
import openai
from dotenv import load_dotenv, find_dotenv
from pynput import keyboard
import sounddevice as sd
import wave
import os
import numpy as np
import whisper
from langchain_openai import OpenAI, ChatOpenAI
from queue import Queue
import io
import soundfile as sf
import threading
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents.agent_types import AgentType

load_dotenv(find_dotenv())
client = openai.Client()


class TalkingLLM():
    def __init__(self, model="gpt-3.5-turbo-0613", whisper_size="small"):
        self.is_recording = False
        self.audio_data = []
        self.samplerate=44100
        self.channels=1
        self.dtype='int16'

        self.whisper = whisper.load_model(whisper_size)
        self.llm = ChatOpenAI(model=model)
        self.llm_queue = Queue()
        self.create_agent()

    def start_or_stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.save_and_transcribe()
            self.audio_data = []
        else:
            print("Starting record")
            self.audio_data = []
            self.is_recording = True

    def create_agent(self):
        agent_prompt_prefix = """
            Você se chama Isaac, e está trabalhando com dataframe pandas no Python. O nome do Dataframe é `df`.
        """

        df = pd.read_csv("df_rent.csv")
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            prefix=agent_prompt_prefix,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    
    def save_and_transcribe(self):
        print("Saving the recording...")
        if "temp.wav" in os.listdir(): os.remove("temp.wav")
        wav_file = wave.open("test.wav", 'wb')
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)  # Corrigido para usar a largura de amostra para int16 diretamente
        wav_file.setframerate(self.samplerate)
        wav_file.writeframes(np.array(self.audio_data, dtype=self.dtype))
        wav_file.close()

        result = self.whisper.transcribe("test.wav", fp16=False)
        print("Usuário:", result["text"])

        # response = self.llm.invoke(result["text"])
        response = self.agent.invoke(result["text"])
        print("AI:", response)
        self.llm_queue.put(response["output"])

        
    def convert_and_play(self):
        tts_text = ''
        while True:
            tts_text += self.llm_queue.get()

            if '.' in tts_text or '?' in tts_text or '!' in tts_text:
                print(tts_text)
                
                spoken_response = client.audio.speech.create(model="tts-1",
                voice='alloy', 
                response_format="opus",
                input=tts_text
                )

                buffer = io.BytesIO()
                for chunk in spoken_response.iter_bytes(chunk_size=4096):
                    buffer.write(chunk)
                buffer.seek(0)

                with sf.SoundFile(buffer, 'r') as sound_file:
                    data = sound_file.read(dtype='int16')
                    sd.play(data, sound_file.samplerate)
                    sd.wait()
                tts_text = ''




    def run(self):
        t1 = threading.Thread(target=self.convert_and_play)
        t1.start()

        def callback(indata, frame_count, time_info, status):
            if self.is_recording:
                self.audio_data.extend(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, 
                            channels=self.channels, 
                            dtype=self.dtype , 
                            callback=callback):
            def on_activate():
                self.start_or_stop_recording()

            def for_canonical(f):
                return lambda k: f(l.canonical(k))

            hotkey = keyboard.HotKey(
                keyboard.HotKey.parse('<cmd>'),
                on_activate)
            with keyboard.Listener(
                    on_press=for_canonical(hotkey.press),
                    on_release=for_canonical(hotkey.release)) as l:
                l.join()


if __name__ == "__main__":
    talking_llm = TalkingLLM()
    talking_llm.run()

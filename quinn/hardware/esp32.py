from __future__ import annotations
from quinn.voice.audio import AudioTransport
class ESP32Transport(AudioTransport):
    """Protocol seam for serial/Wi-Fi ESP32-S3 audio transport; no board pins assumed."""
    async def connect(self)->dict: raise NotImplementedError("Implement ESP32 protocol adapter when hardware is connected.")
    async def disconnect(self)->None: pass
    async def send_audio(self,audio:bytes)->None: raise NotImplementedError
    async def receive_audio(self)->bytes: raise NotImplementedError
    async def send_command(self,command:dict)->dict: raise NotImplementedError
    async def get_status(self)->dict: return {"connected":False,"mock":False}
class MockESP32Transport(ESP32Transport):
    def __init__(self):self.connected=False;self.commands=[]
    async def connect(self):self.connected=True;return await self.get_status()
    async def disconnect(self):self.connected=False
    async def send_audio(self,audio): self.commands.append({"audio_bytes":len(audio)})
    async def receive_audio(self): return b""
    async def send_command(self,command):self.commands.append(command);return {"ok":True,"mock":True}
    async def get_status(self):return {"connected":self.connected,"mock":True,"transport":"local"}

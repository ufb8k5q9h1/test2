from __future__ import annotations
from abc import ABC, abstractmethod
import logging
class MessagingProvider(ABC):
    @abstractmethod
    async def send_message(self,recipient:str,message:str)->dict: ...
class MockMessagingProvider(MessagingProvider):
    async def send_message(self,recipient:str,message:str)->dict:
        logging.getLogger("quinn.tools.messaging").info("mock_message",extra={"recipient":recipient})
        return {"ok":True,"mock":True,"sent":False,"recipient":recipient,"message":"Mock provider logged the message; nothing was sent."}

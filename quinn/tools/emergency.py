from __future__ import annotations
import asyncio, logging
from datetime import datetime, timezone
class EmergencyService:
    def __init__(self, location, messaging, contacts=(), interval_seconds=300):
        self.location,self.messaging,self.contacts,self.interval_seconds=location,messaging,tuple(contacts),interval_seconds; self.active=False; self._task=None; self.events=[]
    def _log(self,event,**extra): self.events.append({"event":event,"at":datetime.now(timezone.utc).isoformat(),**extra}); logging.getLogger("quinn.emergency").warning(event,extra=extra)
    async def activate(self)->dict:
        if self.active:return self.status()
        self.active=True; location=await self.location.get_current_location(); notices=[]
        for contact in self.contacts: notices.append(await self.messaging.send_message(contact,"Quinn emergency mode was activated. Location provider status: "+("available" if location["available"] else "unavailable")))
        self._task=asyncio.create_task(self._periodic_updates()); self._log("emergency_activated",contacts=len(self.contacts))
        return {"ok":True,"active":True,"location":location,"notifications":notices,"mock":True,"message":"Emergency mode activated. Location and messages use safe mock providers."}
    async def _periodic_updates(self):
        try:
            while self.active:
                await asyncio.sleep(self.interval_seconds)
                if self.active: self._log("emergency_location_check",available=(await self.location.get_current_location())["available"])
        except asyncio.CancelledError: pass
    async def stop(self)->dict:
        if self._task:self._task.cancel(); self._task=None
        was=self.active; self.active=False; self._log("emergency_stopped") if was else None
        return {"ok":True,"active":False,"message":"Emergency mode stopped." if was else "Emergency mode was not active."}
    def status(self): return {"ok":True,"active":self.active,"contacts_configured":len(self.contacts),"mock":True}

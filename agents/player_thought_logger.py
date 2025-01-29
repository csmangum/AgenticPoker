import aiofiles
import json
from datetime import datetime
from typing import Any, Dict


class PlayerThoughtLogger:
    def __init__(self, player_name: str, session_id: str):
        self.player_name = player_name
        self.session_id = session_id
        self.file_path = f"logs/players/{session_id}/{player_name}_thoughts.jsonl"

    async def log(self, entry: Dict[str, Any]) -> None:
        async with aiofiles.open(self.file_path, mode='a') as file:
            await file.write(json.dumps(entry) + '\n')

    async def log_prompt(self, prompt: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "type": "prompt",
            "prompt": prompt
        }
        await self.log(entry)

    async def log_response(self, response: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "type": "response",
            "response": response
        }
        await self.log(entry)

    async def log_parsed_action(self, action: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "type": "parsed_action",
            "parsed_action": action
        }
        await self.log(entry)

    async def log_strategy(self, strategy: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "type": "strategy",
            "strategy": strategy
        }
        await self.log(entry)

    async def log_opponent_analysis(self, analysis: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "type": "opponent_analysis",
            "opponent_analysis": analysis
        }
        await self.log(entry)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

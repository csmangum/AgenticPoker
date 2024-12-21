import logging
import os
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class LLMError(Exception):
    """Base class for LLM-related errors."""

    pass


class OpenAIError(LLMError):
    """Errors from OpenAI API."""

    pass


class LocalLLMError(LLMError):
    """Errors from local LLM endpoint."""

    pass


class ResponseParsingError(LLMError):
    """Error parsing LLM response."""

    pass


class ActionType(str, Enum):
    """Valid poker actions."""

    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"


class MessageInterpretation(str, Enum):
    """Valid message interpretations."""

    TRUST = "trust"
    IGNORE = "ignore"
    COUNTER_BLUFF = "counter-bluff"


class StrategyStyle(str, Enum):
    """Valid strategy styles."""

    AGGRESSIVE = "Aggressive Bluffer"
    CAUTIOUS = "Calculated and Cautious"
    CHAOTIC = "Chaotic and Unpredictable"


class PokerAgent:
    """Advanced poker agent with perception, reasoning, communication, and action capabilities.

    A sophisticated AI poker player that combines game state perception, strategic communication,
    and decision-making abilities to play Texas Hold'em poker.

    Attributes:
        name (str): Unique identifier for the agent
        model_type (str): Type of language model to use ('gpt' or 'local_llm')
        last_message (str): Most recent message sent by the agent
        perception_history (list): Historical record of game states and opponent actions
        strategy_style (str): Agent's playing style (e.g., 'Aggressive Bluffer', 'Calculated and Cautious')
    """

    def __init__(
        self,
        name: str,
        model_type: str = "gpt",
        strategy_style: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.last_message = ""
        self.perception_history: List[Dict[str, Any]] = []
        self.strategy_style = strategy_style or random.choice(
            [s.value for s in StrategyStyle]
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)

        # Validate and initialize OpenAI client
        if model_type == "gpt":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
        elif model_type == "local_llm":
            endpoint = os.getenv("LOCAL_LLM_ENDPOINT")
            if not endpoint:
                raise ValueError("LOCAL_LLM_ENDPOINT environment variable not set")
            self.endpoint = endpoint

    def perceive(
        self, game_state: Dict[str, Any], opponent_message: str
    ) -> Dict[str, Any]:
        """Process and store current game state and opponent's message.

        Args:
            game_state (dict): Current state of the poker game
            opponent_message (str): Message received from the opponent

        Returns:
            dict: Perception data including game state, opponent message, and timestamp
        """
        perception = {
            "game_state": game_state,
            "opponent_message": opponent_message,
            "timestamp": time.time(),
        }
        self.perception_history.append(perception)
        return perception

    def get_message(self, game_state: Dict[str, Any]) -> str:
        """Generate strategic communication based on game state and strategy style.

        Uses LLM to create contextually appropriate messages that align with the agent's
        strategy style and current game situation.

        Args:
            game_state (dict): Current state of the poker game

        Returns:
            str: Strategic message to influence opponent
        """
        prompt = f"""
        You are a {self.strategy_style} poker player in Texas Hold'em.
        
        Current situation:
        Game State: {game_state}
        Your recent observations: {self.perception_history[-3:] if len(self.perception_history) > 0 else "None"}
        
        Generate a strategic message to influence your opponent. Your personality is {self.strategy_style}.
        
        Your message should:
        1. Match your strategy style
        2. Be under 10 words
        3. Try to influence your opponent's next decision
        4. Consider your previous interactions
        
        What message will you send?
        """
        self.last_message = self._query_llm(prompt).strip()
        return self.last_message

    def interpret_message(self, opponent_message: str) -> str:
        """Enhanced message interpretation with historical context.

        Analyzes opponent messages considering recent game history and agent's strategy style.

        Args:
            opponent_message (str): Message received from the opponent

        Returns:
            str: Interpretation result ('trust', 'ignore', or 'counter-bluff')
        """
        recent_history = self.perception_history[-3:] if self.perception_history else []

        prompt = f"""
        You are a {self.strategy_style} poker player.
        Opponent's message: '{opponent_message}'
        Recent game history: {recent_history}
        
        Based on your strategy style and the game history:
        1. Analyze if they are bluffing, truthful, or misleading
        2. Consider their previous behavior patterns
        3. Think about how this fits your strategy style
        
        Respond with only: 'trust', 'ignore', or 'counter-bluff'
        """
        return self._query_llm(prompt).strip().lower()

    def _normalize_action(self, action: str) -> str:
        """Normalize the LLM's action response to a valid action."""
        # Remove any quotes and extra whitespace
        action = action.lower().strip().strip("'\"")

        # Extract just the action word if it's embedded in a sentence
        action_words = {
            "fold": "fold",
            "call": "call",
            "raise": "raise",
            "check": "call",  # normalize check to call
            "bet": "raise",  # normalize bet to raise
        }

        # First try exact match
        if action in action_words:
            return action_words[action]

        # Then look for action words in the response
        for word in action.split():
            word = word.strip(".:,!?*()[]'\"")  # Remove punctuation and quotes
            if word in action_words:
                return action_words[word]

        # If no valid action found, log and return None
        self.logger.warning("Could not parse action from LLM response: '%s'", action)
        return None

    def get_action(
        self, game_state: Dict[str, Any], opponent_message: Optional[str] = None
    ) -> str:
        """Get action with improved parsing and validation."""
        try:
            prompt = f"""
            You are a {self.strategy_style} poker player in a crucial moment.
            
            Current situation:
            Game State: {game_state}
            Opponent's Message: '{opponent_message or "nothing"}'
            Recent History: {self.perception_history[-3:] if self.perception_history else []}
            
            Consider:
            1. Your strategy style: {self.strategy_style}
            2. The opponent's recent behavior
            3. Your position and chip stack
            4. The credibility of their message
            
            Important: Respond with exactly one word, without quotes: fold, call, or raise
            """

            raw_action = self._query_llm(prompt)
            action = self._normalize_action(raw_action)

            if action is None:
                self.logger.warning(
                    "LLM returned invalid action '%s', falling back to 'fold'",
                    raw_action,
                )
                return "fold"

            return action

        except LLMError as e:
            self.logger.error("LLM error in get_action: %s", str(e))
            return "fold"

    def _query_gpt(self, prompt: str) -> str:
        """Query OpenAI's GPT model with error handling."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a {self.strategy_style} poker player.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=20,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise OpenAIError(f"GPT query failed: {str(e)}") from e

    def _query_local_llm(self, prompt: str) -> str:
        """Query local LLM endpoint with error handling."""
        try:
            response = requests.post(
                self.endpoint,
                json={"prompt": prompt, "max_tokens": 20},
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result["choices"]:
                raise LocalLLMError("Invalid response format from local LLM")

            return result["choices"][0]["text"]

        except requests.RequestException as e:
            raise LocalLLMError(f"Local LLM request failed: {str(e)}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise LocalLLMError(f"Invalid response from local LLM: {str(e)}") from e

    def _query_llm(self, prompt: str) -> str:
        """Enhanced LLM query with retries and comprehensive error handling."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    "[LLM Query] Attempt %d for %s", attempt + 1, self.name
                )
                self.logger.debug("[LLM Query] Prompt: %s", prompt)

                if self.model_type == "gpt":
                    result = self._query_gpt(prompt)
                else:
                    result = self._query_local_llm(prompt)

                self.logger.info("[LLM Query] Response: %s", result)
                return result

            except (OpenAIError, LocalLLMError) as e:
                last_error = e
                self.logger.error(
                    "[LLM Query] %s error on attempt %d: %s",
                    self.model_type,
                    attempt + 1,
                    str(e),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

        self.logger.error(
            "[LLM Query] All %d attempts failed for %s", self.max_retries, self.name
        )
        raise LLMError(f"Failed after {self.max_retries} attempts") from last_error

    def update_strategy(self, game_outcome: Dict[str, Any]) -> None:
        """Update agent's strategy based on game outcomes and performance.

        Analyzes game results to potentially adjust strategy style and decision-making
        patterns for future games.

        Args:
            game_outcome (dict): Results and statistics from the completed game
        """
        prompt = f"""
        You are a poker player analyzing your performance.
        
        Game outcome: {game_outcome}
        Current strategy: {self.strategy_style}
        Recent history: {self.perception_history[-5:] if self.perception_history else "None"}
        
        Should you:
        1. Keep current strategy: {self.strategy_style}
        2. Switch to "Aggressive Bluffer"
        3. Switch to "Calculated and Cautious"
        4. Switch to "Chaotic and Unpredictable"
        
        Respond with just the number (1-4).
        """

        response = self._query_llm(prompt).strip()

        strategy_map = {
            "2": "Aggressive Bluffer",
            "3": "Calculated and Cautious",
            "4": "Chaotic and Unpredictable",
        }

        if response in strategy_map:
            self.logger.info(
                "[Strategy Update] %s changing strategy from %s to %s",
                self.name,
                self.strategy_style,
                strategy_map[response],
            )
            self.strategy_style = strategy_map[response]

    def analyze_opponent(self) -> Dict[str, Any]:
        """Analyze opponent's behavior patterns and tendencies.

        Reviews perception history to identify patterns in opponent's actions,
        messages, and betting behavior.

        Returns:
            dict: Analysis results including behavior patterns and threat assessment
        """
        if not self.perception_history:
            return {"patterns": "insufficient data", "threat_level": "unknown"}

        prompt = f"""
        Analyze this opponent's behavior patterns:
        Recent history: {self.perception_history[-5:]}
        
        Provide a concise analysis in this exact JSON format:
        {{
            "patterns": "<one word>",
            "threat_level": "<low/medium/high>"
        }}
        """

        try:
            response = self._query_llm(prompt).strip()
            # Basic validation that it's in the expected format
            if '"patterns"' in response and '"threat_level"' in response:
                return eval(
                    response
                )  # Safe here since we control the LLM output format
        except Exception as e:
            self.logger.error(
                "[Opponent Analysis] Error parsing LLM response: %s", str(e)
            )

        return {"patterns": "unknown", "threat_level": "medium"}

    def reset_state(self) -> None:
        """Reset agent's state for a new game.

        Clears perception history while maintaining strategy style and name.
        """
        self.perception_history = []
        self.last_message = ""
        self.logger.info("[Reset] Agent %s reset for new game", self.name)

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve agent's performance statistics and current state.

        Returns:
            dict: Statistics including strategy style, perception history length,
                  and other relevant metrics
        """
        return {
            "name": self.name,
            "strategy_style": self.strategy_style,
            "perception_history_length": len(self.perception_history),
            "model_type": self.model_type,
            "last_message": self.last_message,
        }
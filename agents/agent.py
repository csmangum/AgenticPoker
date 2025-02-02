import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from agents.llm_client import LLMClient
from agents.llm_response_generator import LLMResponseGenerator
from agents.memory import ChromaMemoryStore
from agents.prompts import DISCARD_PROMPT
from agents.strategy_planner import StrategyPlanner
from config import GameConfig
from data.model import Game
from data.types.action_decision import ActionDecision, ActionType
from data.types.discard_decision import DiscardDecision
from game.evaluator import HandEvaluation
from game.player import Player
from game.utils import get_min_bet, validate_bet_amount
from loggers.agent_logger import AgentLogger

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")


class Agent(Player):
    """An intelligent poker agent that uses LLM-based decision making and various cognitive modules."""

    def __init__(
        self,
        name: str,
        chips: int = 1000,
        strategy_style: str = "Aggressive Bluffer",  #! make a play style model
        use_reasoning: bool = True,
        use_reflection: bool = True,
        use_planning: bool = True,
        use_opponent_modeling: bool = True,
        use_reward_learning: bool = True,
        learning_rate: float = 0.1,  #! is this needed?
        config: GameConfig = None,  #! is this needed?
        session_id: str = None,  #! is this needed?
        communication_style: str = "Intimidating",
    ):
        super().__init__(name, chips)
        self.config = config
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_planning = use_planning
        self.use_opponent_modeling = use_opponent_modeling
        self.use_reward_learning = use_reward_learning
        self.learning_rate = learning_rate

        self.last_message = ""
        self.last_opponent_action = None
        self.perception_history: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.strategy_style = strategy_style
        self.communication_style = communication_style
        self.emotional_state = "confident"
        self.table_history = []
        self.max_retries = 3
        self.retry_delay = 1.0
        self.personality_traits = {
            "aggression": 0.5,
            "bluff_frequency": 0.5,
            "risk_tolerance": 0.5,
        }

        # Initialize LLM client first
        self.llm_client = LLMClient(api_key=API_KEY, model="gpt-3.5-turbo")

        # Then initialize strategy planner with llm_client
        if self.use_planning:
            self.strategy_planner = StrategyPlanner(
                strategy_style=self.strategy_style,
                plan_duration=30.0,
            )
        else:
            self.strategy_planner = None

        # Initialize memory store with session-specific collection name
        collection_name = f"agent_{name.lower().replace(' ', '_')}_{session_id}_memory"
        self.memory_store = ChromaMemoryStore(collection_name)

        # Keep short-term memory in lists for immediate context
        self.short_term_limit = 3
        self.perception_history: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []

        # Initialize opponent modeling structures only if enabled
        if self.use_opponent_modeling:
            self.opponent_stats = defaultdict(
                lambda: {
                    "actions": defaultdict(int),
                    "bet_sizes": [],
                    "showdown_hands": [],
                    "bluff_attempts": 0,
                    "bluff_successes": 0,
                    "fold_to_raise_count": 0,
                    "raise_faced_count": 0,
                    "last_five_actions": [],
                    "position_stats": defaultdict(lambda: defaultdict(int)),
                }
            )
            self.opponent_models = {}

        # Initialize action history
        self.action_history = []

        # Initialize reward learning structures if enabled
        if self.use_reward_learning:
            self.action_history: List[Tuple[str, Dict[str, Any], int]] = (
                []
            )  # (action, state, reward)
            self.reward_weights = {
                "chip_gain": 1.0,
                "win_rate": 0.8,
                "bluff_success": 0.6,
                "position_value": 0.4,
            }
            self.action_values = {"fold": 0.0, "call": 0.0, "raise": 0.0}

        # Initialize session tracking
        self.current_hand_id = None
        self.current_round_id = None
        self.session_start_time = time.time()

        # Track important game events
        self.last_action = None
        self.last_position = None
        self.starting_chips = chips
        self.chip_history = []

    def close(self):
        """Clean up external resources explicitly.

        Ensures proper cleanup of:
        1. In-memory data structures (perception history, conversation history, plans)
        2. Memory store connection (ChromaDB)
        3. OpenAI client

        This method should be called explicitly when done with the agent, or used
        via the context manager pattern:

        Example:
            # Method 1: Explicit cleanup
            agent = LLMAgent(name="Bot1")
            try:
                # Use agent...
            finally:
                agent.close()

            # Method 2: Context manager (preferred)
            with LLMAgent(name="Bot2") as agent:
                # Use agent...
                # Cleanup happens automatically
        """
        try:
            # Clear in-memory data
            if hasattr(self, "perception_history"):
                self.perception_history.clear()
                del self.perception_history

            if hasattr(self, "conversation_history"):
                self.conversation_history.clear()
                del self.conversation_history

            if hasattr(self, "strategy_planner"):
                self.strategy_planner.current_plan = None
                del self.strategy_planner.current_plan

            if hasattr(self, "opponent_stats"):
                self.opponent_stats.clear()
                del self.opponent_stats

            if hasattr(self, "opponent_models"):
                self.opponent_models.clear()
                del self.opponent_models

            # Clean up memory store
            if hasattr(self, "memory_store"):
                try:
                    self.memory_store.close()
                except Exception as e:
                    AgentLogger.log_cleanup_error(e, "memory store")
                finally:
                    del self.memory_store

            # Clean up OpenAI client
            if hasattr(self, "client"):
                try:
                    del self.client
                except Exception as e:
                    AgentLogger.log_cleanup_error(e, "OpenAI client")

        except Exception as e:
            AgentLogger.log_general_cleanup_error(e)

    def __enter__(self):
        """Context manager entry point.

        Enables the agent to be used with Python's 'with' statement for automatic
        resource cleanup:

            with LLMAgent(name="Bot") as agent:
                # Agent is automatically cleaned up after this block

        Returns:
            LLMAgent: The agent instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point.

        Called automatically when exiting a 'with' block. Ensures cleanup happens
        even if an exception occurs within the block.

        Args:
            exc_type: Type of exception that occurred (if any)
            exc_val: Exception instance (if any)
            exc_tb: Exception traceback (if any)

        Note:
            Exceptions are not suppressed and will propagate after cleanup.
        """
        self.close()

    def __del__(self):
        """Fallback cleanup if close() wasn't called explicitly.

        This is a safety net for cases where the agent wasn't properly closed.
        However, it's better to use either:
        1. Explicit close() calls
        2. Context manager ('with' statement)

        Note:
            Errors are suppressed here since this may be called during interpreter
            shutdown when some resources are already gone.
        """
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during interpreter shutdown

    def decide_action(self, game: "Game") -> ActionDecision:
        """Determine the next poker action based on the current game state."""
        # Get hand evaluation before making decision
        hand_eval: HandEvaluation = self.hand.evaluate() if self.hand else None

        # Plan strategy if strategy planner is enabled
        if self.use_planning:
            self.strategy_planner.plan_strategy(self, game, hand_eval)

        decided_action = self._decide_action(game, hand_eval)

        return decided_action

    def _decide_action(
        self,
        game: "Game",
        hand_eval: Optional[HandEvaluation] = None,
    ) -> ActionDecision:
        """Execute an action based on current plan and game state."""
        try:
            current_plan = (
                self.strategy_planner.current_plan if self.use_planning else None
            )
            # Delegate action creation to the strategy generator
            action: ActionDecision = LLMResponseGenerator.generate_action(
                player=self,
                game=game,
                current_plan=current_plan,
                hand_eval=hand_eval,
            )

            # Validate raise amount if it's a raise action
            if action.action_type == ActionType.RAISE:
                min_bet = get_min_bet(game)
                action.raise_amount = validate_bet_amount(action.raise_amount, min_bet)

            AgentLogger.log_action(action)
            return action

        except Exception as e:
            AgentLogger.log_action(None, error=e)
            return ActionDecision(
                action_type=ActionType.CALL, reasoning="Failed to decide action"
            )

    def get_message(self, game) -> str:
        """Generate table talk using LLM.

        Args:
            game: Current game state

        Returns:
            str: A message for table talk, or empty string if generation fails
        """
        try:
            # Create message prompt
            game_state = game.get_state()
            prompt = self._create_message_prompt(game_state)

            system_message = (
                f"You are a {self.communication_style} {self.strategy_style} "
                f"poker player engaging in table talk."
            )

            # Query LLM with lower temperature for more consistent messaging
            response = self.llm_client.query(
                prompt=prompt,
                temperature=0.5,
                system_message=system_message,
                tags=["table_talk"],
            )

            # Parse response - look for MESSAGE: prefix
            if "MESSAGE:" not in response:
                AgentLogger.log_message_generation()
                return "..."  # Return a default message instead of empty string

            # Extract the message part after MESSAGE:
            message_line = next(
                line.strip()
                for line in response.split("\n")
                if line.strip().startswith("MESSAGE:")
            )
            message = message_line.replace("MESSAGE:", "").strip()

            return message

        except Exception as e:
            AgentLogger.log_message_generation(error=e)
            return "..."  # Return a default message on error

    def _create_message_prompt(self, game_state: str) -> str:
        """Create prompt for message generation.

        Args:
            game_state: Current state of the game

        Returns:
            str: Formatted prompt for the LLM
        """
        return f"""
        You are a {self.communication_style} poker player.
        Current emotional state: {self.emotional_state}
        Game state: {game_state}
        Recent table history: {self.table_history[-3:] if self.table_history else 'None'}
        
        Generate a short, natural poker table message that:
        1. Fits your communication style ({self.communication_style})
        2. Responds to the current game state
        3. Maintains character consistency
        
        Respond with just the message in this format:
        MESSAGE: <your message here>
        """

    def perceive(self, game_state: str, opponent_message: Optional[str] = None) -> Dict:
        #! validate this
        #! this is how player gets a state for further processing
        """Enhanced perception with communication context.

        Args:
            game_state (str): Current state of the game
            opponent_message (Optional[str]): Message from opponent, if any

        Returns:
            Dict: Enhanced perception data including memory updates
        """
        # Get base perception from parent class
        perception = super().perceive(game_state, opponent_message)

        # Add opponent message to table history if present
        if opponent_message:
            self.table_history.append(f"Opponent: {opponent_message}")
            # Trim history if too long
            if len(self.table_history) > 10:
                self.table_history = self.table_history[-10:]

        # Store perception in memory
        self.memory_store.add_memory(
            text=f"Game State: {game_state}"
            + (f"\nOpponent: {opponent_message}" if opponent_message else ""),
            metadata={"type": "perception", "timestamp": perception["timestamp"]},
        )

        return perception

    def decide_discard(
        self, game_state: Optional[Dict[str, Any]] = None
    ) -> DiscardDecision:
        """Decide which cards to discard."""
        try:
            discard: DiscardDecision = LLMResponseGenerator.generate_discard(
                self, game_state, self.hand.cards
            )

            return discard

        except Exception as e:
            AgentLogger.log_discard_error(e)
            return DiscardDecision(
                discard_indices=[], reasoning="Failed to decide discard"
            )

    def update_strategy(self, game_outcome: Dict[str, Any]) -> None:
        #! need to refactor and combine with strategy_planner (with strategy_manager)
        """Update agent's strategy based on game outcomes and performance.

        Analyzes game results to potentially adjust strategy style and decision-making
        patterns for future games. Uses LLM to evaluate strategy effectiveness.

        Args:
            game_outcome (dict): Results and statistics from the completed game

        Note:
            - Can switch between predefined strategy styles
            - Considers recent history in decision
            - Logs strategy changes for monitoring
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

        response = self.llm_client.query(prompt, tags=["strategy_update"]).strip()

        strategy_map = {
            "2": "Aggressive Bluffer",
            "3": "Calculated and Cautious",
            "4": "Chaotic and Unpredictable",
        }

        if response in strategy_map:
            AgentLogger.log_strategy_update(
                self.name, self.strategy_style, strategy_map[response]
            )
            self.strategy_style = strategy_map[response]

    def analyze_opponent(self, opponent_name: str, game_state: str) -> Dict[str, Any]:
        #! validate and refactor this
        """Enhanced opponent analysis using historical data and LLM interpretation.

        Analyzes opponent patterns considering:
        - Action frequencies
        - Betting patterns
        - Position-based tendencies
        - Bluffing frequency
        - Response to aggression
        """
        # Default analysis structure
        default_analysis = {
            "patterns": "unknown",
            "threat_level": "medium",
            "style": "unknown",
            "weaknesses": [],
            "strengths": [],
            "recommended_adjustments": [],
        }

        # Return default if opponent modeling is disabled
        if not self.use_opponent_modeling:
            return default_analysis

        stats = self.opponent_stats[opponent_name]

        # Return default if insufficient data
        total_actions = sum(stats["actions"].values())
        if total_actions == 0:
            default_analysis.update(
                {
                    "patterns": "insufficient data",
                    "style": "unknown",
                    "threat_level": "unknown",
                }
            )
            return default_analysis

        # Calculate key metrics
        aggression_frequency = (
            (stats["actions"]["raise"] / total_actions) if total_actions > 0 else 0
        )
        fold_to_raise_ratio = (
            (stats["fold_to_raise_count"] / stats["raise_faced_count"])
            if stats["raise_faced_count"] > 0
            else 0
        )
        bluff_success_rate = (
            (stats["bluff_successes"] / stats["bluff_attempts"])
            if stats["bluff_attempts"] > 0
            else 0
        )

        # Prepare statistical summary for LLM
        stats_summary = f"""
        Opponent Analysis for {opponent_name}:
        - Action Distribution: {dict(stats['actions'])}
        - Average Bet Size: ${np.mean(stats['bet_sizes']) if stats['bet_sizes'] else 0:.2f}
        - Fold to Raise: {fold_to_raise_ratio:.2%}
        - Bluff Success Rate: {bluff_success_rate:.2%}
        - Recent Actions: {stats['last_five_actions'][-5:]}
        - Position Tendencies: {dict(stats['position_stats'])}
        """

        prompt = f"""
        Analyze this poker opponent's playing style and patterns:
        
        {stats_summary}
        Current Game State: {game_state}
        
        Provide a detailed analysis in this exact JSON format:
        {{
            "patterns": "<primary pattern>",
            "threat_level": "<low/medium/high>",
            "style": "<tight-aggressive/loose-passive/unknown>",
            "weaknesses": ["<exploitable pattern>"],
            "strengths": ["<strong pattern>"],
            "recommended_adjustments": ["<strategic adjustment>"]
        }}
        
        Base the analysis on:
        1. Position-based tendencies
        2. Betting patterns and sizes
        3. Response to aggression
        4. Bluffing frequency and success
        5. Recent behavior changes
        
        Always include all fields, use "unknown" for uncertain values.
        """

        try:
            response = self.llm_client.query(prompt, tags=["opponent_analysis"])
            analysis = eval(response.strip())  # Safe since we control LLM output format

            # Ensure all required keys are present
            for key in default_analysis:
                if key not in analysis:
                    analysis[key] = default_analysis[key]

            # Cache analysis for future reference
            self.opponent_models[opponent_name] = {
                "last_analysis": analysis,
                "timestamp": time.time(),
                "confidence": min(
                    total_actions / 20, 1.0
                ),  # Confidence based on sample size
            }

            return analysis

        except Exception as e:
            AgentLogger.log_opponent_analysis_error(e)
            return default_analysis

    def __str__(self):
        return f"Agent(name={self.name}, strategy={self.strategy_style})"

    def __repr__(self):
        return f"Agent(name={self.name}, strategy={self.strategy_style})"

    def record_game_state(self, game: "Game") -> None:
        """Record current game state in memory."""
        # Convert GameState to a string representation
        game_state = game.get_state()
        state_text = (
            f"Round {game_state.round_state.round_number} - "
            f"Phase: {game_state.round_state.phase}, "
            f"Players: {len(game_state.players)}, "
            f"Pot: ${game_state.pot_state.total_pot}, "
            f"Current bet: ${game_state.round_state.current_bet}"
        )

        self.memory_store.add_memory(
            text=state_text,
            metadata={
                "memory_type": "game_state",
                "hand_id": self.current_hand_id,
                "round_id": self.current_round_id,
                "position": self.position,
                "chips": self.chips,
                "pot_size": game.pot.pot,
            },
        )

    def record_action(self, action: ActionDecision, game: "Game") -> None:
        """Record an action taken by the agent."""
        self.memory_store.add_memory(
            text=str(action),
            metadata={
                "memory_type": "game_action",
                "hand_id": self.current_hand_id,
                "round_id": self.current_round_id,
                "action_type": action.action_type.value,
                "chips_delta": (
                    -action.raise_amount if hasattr(action, "raise_amount") else 0
                ),
                "position": self.position,
            },
        )

    def record_opponent_action(
        self, opponent_name: str, action: str, bet_amount: Optional[int] = None
    ) -> None:
        """Record an opponent's action."""
        self.memory_store.add_memory(
            text=f"{opponent_name}: {action}"
            + (f" ${bet_amount}" if bet_amount else ""),
            metadata={
                "memory_type": "opponent_action",
                "hand_id": self.current_hand_id,
                "round_id": self.current_round_id,
                "opponent_name": opponent_name,
                "action": action,
                "bet_amount": bet_amount if bet_amount is not None else 0,  # Default to 0 instead of None
            },
        )

    def record_hand_result(
        self, won: bool, amount: int, opponent_cards: Optional[str] = None
    ) -> None:
        """Record the result of a completed hand."""
        self.memory_store.add_memory(
            text=f"Hand result: {'Won' if won else 'Lost'} ${amount}"
            + (f" | Opponent showed: {opponent_cards}" if opponent_cards else ""),
            metadata={
                "memory_type": "hand_result",
                "hand_id": self.current_hand_id,
                "won": won,
                "chips_delta": amount,
                "opponent_cards": str(opponent_cards) if opponent_cards else "",  # Convert to empty string instead of None
            },
        )

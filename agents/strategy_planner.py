import time
from typing import TYPE_CHECKING, Optional

from data.types.plan import Approach, BetSizing, Plan
from game.evaluator import HandEvaluation
from loggers.strategy_logger import StrategyLogger

from .llm_response_generator import LLMResponseGenerator
from .reasoning_module import ReasoningModule

if TYPE_CHECKING:
    from game.game import Game
    from game.player import Player


DEFAULT_PLAN_DURATION = 30.0
REPLAN_STACK_THRESHOLD = 100


class StrategyPlanner:
    """Strategic planning and execution engine for poker agents.

    This class manages the complete strategic decision-making process for poker agents,
    including plan generation, validation, execution, and automatic renewal based on
    game conditions. It uses LLM-based decision making through an LLMClient for both
    strategy planning and action execution.

    Attributes:
        strategy_style (str): The playing style used for planning (e.g., 'aggressive', 'conservative')
        plan_duration (float): Duration in seconds for which a plan remains valid
        REPLAN_STACK_THRESHOLD (int): Stack size change that triggers a replan
        current_plan (Optional[Plan]): The currently active strategic plan
        last_metrics (Optional[dict]): Last recorded game metrics used for planning
        reasoning_module (ReasoningModule): The reasoning module for enhanced strategy planning
    """

    def __init__(
        self,
        strategy_style: str,
        plan_duration: float = DEFAULT_PLAN_DURATION,
        replan_threshold: int = REPLAN_STACK_THRESHOLD,
    ) -> None:
        """Initialize the strategy planner with configuration parameters.

        Args:
            strategy_style (str): Playing style to use for planning (e.g., 'aggressive', 'conservative')
            plan_duration (float, optional): How long plans remain valid in seconds. Defaults to DEFAULT_PLAN_DURATION.
            replan_threshold (int, optional): Stack change threshold that triggers replanning.
                Defaults to REPLAN_STACK_THRESHOLD.
        """
        self.strategy_style = strategy_style
        self.plan_duration = plan_duration
        self.REPLAN_STACK_THRESHOLD = replan_threshold
        self.current_plan = None
        self.last_metrics = None
        self.reasoning_module = ReasoningModule()

    def plan_strategy(
        self,
        player: "Player",
        game: "Game",
        hand_eval: Optional[HandEvaluation] = None,
    ) -> None:
        """Generate or update the agent's strategic plan based on current game state.

        This method evaluates the current game situation and either generates a new plan
        or maintains the existing one based on replanning criteria. It uses LLM-generated
        responses to create sophisticated playing strategies.

        Args:
            player (Player): The player for whom to generate the strategy
            game (Game): Current game state including all relevant poker information
            hand_eval (Optional[HandEvaluation], optional): Pre-computed hand evaluation.
                Defaults to None.

        Raises:
            Exception: If plan generation fails, falls back to default plan
        """
        try:
            if self.current_plan and not self.requires_replanning(game, player):
                StrategyLogger.log_plan_reuse(self.current_plan)
                return  # Early return when reusing existing plan

            # Use the strategy generator to get new plan data
            plan_data = LLMResponseGenerator.generate_plan(
                player=player,
                game_state=game.get_state(),
                hand_eval=hand_eval,
            )
            self.current_plan = self._create_plan_from_response(plan_data)
            StrategyLogger.log_new_plan(self.current_plan)

        except Exception as e:
            StrategyLogger.log_plan_error(e)
            self.current_plan = self._create_default_plan()

    def _create_default_plan(self) -> Plan:
        """Create a default Plan object when errors occur or no plan is available.

        Generates a balanced, conservative plan with standard thresholds as a fallback
        mechanism when normal plan generation fails.

        Returns:
            Plan: A balanced default plan with standard thresholds and settings:
                - Balanced approach
                - Medium bet sizing
                - 0.5 bluff threshold
                - 0.3 fold threshold
                - Default duration expiry
        """
        return Plan(
            approach=Approach.BALANCED,
            reasoning="Default fallback plan due to error",
            bet_sizing=BetSizing.MEDIUM,
            bluff_threshold=0.5,
            fold_threshold=0.3,
            expiry=time.time() + DEFAULT_PLAN_DURATION,
            adjustments=[],
            target_opponent=None,
        )

    def _create_plan_from_response(self, plan_data: dict) -> Plan:
        """Create a Plan object from LLM response data with validation.

        Converts the raw dictionary response from the LLM into a structured Plan object,
        applying necessary validations and default values where needed.

        Args:
            plan_data (dict): Parsed response data from LLM containing:
                - approach: Strategy approach (e.g., 'aggressive', 'balanced')
                - reasoning: Explanation of the strategic choices
                - bet_sizing: Betting size preference
                - bluff_threshold: Threshold for bluffing decisions
                - fold_threshold: Threshold for folding decisions

        Returns:
            Plan: A new validated plan object with all required fields populated

        Note:
            Default values are applied for any missing fields in the plan_data
        """
        return Plan(
            approach=Approach(plan_data.get("approach", "balanced")),
            reasoning=plan_data.get("reasoning", "Default reasoning"),
            bet_sizing=BetSizing(plan_data.get("bet_sizing", "medium")),
            bluff_threshold=float(plan_data.get("bluff_threshold", 0.5)),
            fold_threshold=float(plan_data.get("fold_threshold", 0.3)),
            expiry=time.time() + DEFAULT_PLAN_DURATION,
            adjustments=[],
            target_opponent=None,
        )

    def requires_replanning(self, game: "Game", player: "Player") -> bool:
        """Determine if current game state requires a new strategic plan.

        Evaluates various game conditions to decide if a new plan should be generated,
        including plan expiration and significant changes in game state.

        Args:
            game (Game): Current game state to evaluate
            player (Player): Player whose plan needs evaluation

        Returns:
            bool: True if replanning is required, False otherwise

        Note:
            Currently checks for plan existence and expiration.
            TODO: Implement additional checks for stack size changes and opponent behavior
        """
        # Always replan if no current plan exists
        if not self.current_plan:
            StrategyLogger.log_planning_check(
                "No current plan exists - replanning required"
            )
            return True

        try:
            # Check if plan has expired
            if self.current_plan.is_expired():
                StrategyLogger.log_planning_check(
                    "Current plan has expired - replanning required"
                )
                return True

            return False

        except Exception as e:
            StrategyLogger.log_replan_error(e)
            return False  # Safe fallback - keep current plan on error

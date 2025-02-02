from agents.prompts import ACTION_PROMPT, DISCARD_PROMPT, PLANNING_PROMPT
from data.types.action_decision import ActionDecision
from data.types.discard_decision import DiscardDecision
from data.types.llm_responses import PlanResponse
from data.types.plan import Plan


class LLMResponseGenerator:
    """
    Encapsulates the logic for generating prompts, querying the LLM,
    and parsing the responses for poker strategy and actions.
    """

    @classmethod
    def generate_plan(cls, player, game_state, hand_eval) -> "PlanResponse":
        """
        Create a plan by calling the LLM with the appropriate planning prompt.
        Returns the parsed dictionary of plan data.
        """
        prompt = PLANNING_PROMPT.format(
            strategy_style=player.strategy_style,
            game_state=game_state,
            hand_eval=hand_eval,
        )
        response = player.llm_client.query(
            prompt=prompt, temperature=0.7, max_tokens=200, tags=["planning"]
        )
        return PlanResponse.parse_llm_response(response)

    @classmethod
    def generate_action(
        cls, player, game, current_plan: Plan, hand_eval
    ) -> "ActionDecision":
        """
        Create an action by calling the LLM with the action prompt.
        Returns the raw LLM response string for further parsing.
        """
        # Set default values if planning is disabled
        plan_approach = getattr(current_plan, "approach", "No specific approach")
        plan_reasoning = getattr(current_plan, "reasoning", "Direct decision making")
        bluff_threshold = getattr(current_plan, "bluff_threshold", 0.5)
        fold_threshold = getattr(current_plan, "fold_threshold", 0.7)

        # Calculate betting ranges
        current_bet = game.table.current_bet
        min_bet = game.config.min_bet
        min_raise = current_bet + min_bet
        max_raise = min(player.chips, current_bet + player.chips)

        # Determine if this is pre-draw phase
        is_pre_draw = game.round_state.phase == "PRE_DRAW"
        pre_draw_note = (
            "(Note: This is the pre-draw betting round. You will have a chance to discard and draw new cards after this betting round.)"
            if is_pre_draw
            else ""
        )

        execution_prompt = ACTION_PROMPT.format(
            strategy_style=player.strategy_style,
            game_state=game.get_state(),
            hand_eval=hand_eval,
            plan_approach=plan_approach,
            plan_reasoning=plan_reasoning,
            bluff_threshold=bluff_threshold,
            fold_threshold=fold_threshold,
            pre_draw_note=pre_draw_note,
            min_raise=min_raise,
            max_raise=max_raise,
            current_bet=current_bet
        )
        response = player.llm_client.query(
            prompt=execution_prompt,
            temperature=0.7,
            max_tokens=100,
            tags=["action_generation"],
        )
        return ActionDecision.parse_llm_response(response)

    @classmethod
    def generate_discard(cls, player, game_state, cards) -> "DiscardDecision":
        """Create a discard decision by calling the LLM with the discard prompt.

        Args:
            player: The player making the discard decision
            game_state: Current state of the game
            cards: List of 5 card objects representing current hand

        Returns:
            DiscardDecision: Parsed and validated discard decision

        Raises:
            ValueError: If LLM response cannot be parsed into a valid discard decision
        """
        prompt = DISCARD_PROMPT.format(
            strategy_style=player.strategy_style,
            game_state=game_state,
            cards=cards,
        )

        response = player.llm_client.query(
            prompt=prompt, temperature=0.7, max_tokens=100, tags=["discard_generation"]
        )

        return DiscardDecision.parse_llm_response(response)

    @classmethod
    def generate_reasoning_prompt(cls, reasoning_type: str, context: str) -> str:
        """
        Generate a reasoning prompt based on the type of reasoning and context provided.
        """
        return f"{reasoning_type.capitalize()} reasoning: {context}"

    @classmethod
    def parse_reasoning_response(cls, response: str) -> str:
        """
        Parse the reasoning response from the LLM.
        """
        return response.strip()

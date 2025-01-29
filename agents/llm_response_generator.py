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
    async def generate_plan(cls, player, game_state, hand_eval) -> "PlanResponse":
        """
        Create a plan by calling the LLM with the appropriate planning prompt.
        Returns the parsed dictionary of plan data.
        """
        prompt = PLANNING_PROMPT.format(
            strategy_style=player.strategy_style,
            game_state=game_state,
            hand_eval=hand_eval,
        )
        response = await player.llm_client.query_async(
            prompt=prompt, temperature=0.7, max_tokens=200
        )
        plan_response = PlanResponse.parse_llm_response(response)

        # Log the planning process
        await player.thought_logger.log_prompt(prompt)
        await player.thought_logger.log_response(response)
        await player.thought_logger.log_strategy(plan_response)

        return plan_response

    @classmethod
    async def generate_action(
        cls, player, game_state, current_plan: Plan, hand_eval
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

        execution_prompt = ACTION_PROMPT.format(
            strategy_style=player.strategy_style,
            game_state=game_state,
            hand_eval=hand_eval,
            plan_approach=plan_approach,
            plan_reasoning=plan_reasoning,
            bluff_threshold=bluff_threshold,
            fold_threshold=fold_threshold,
        )
        response = await player.llm_client.query_async(
            prompt=execution_prompt,
            temperature=0.7,
            max_tokens=100,
        )
        action_decision = ActionDecision.parse_llm_response(response)

        # Log the decision-making process
        await player.thought_logger.log_prompt(execution_prompt)
        await player.thought_logger.log_response(response)
        await player.thought_logger.log_parsed_action(action_decision)

        return action_decision

    @classmethod
    async def generate_discard(cls, player, game_state, cards) -> "DiscardDecision":
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

        response = await player.llm_client.query_async(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
        )

        discard_decision = DiscardDecision.parse_llm_response(response)

        # Log the discard decision process
        await player.thought_logger.log_prompt(prompt)
        await player.thought_logger.log_response(response)
        await player.thought_logger.log_parsed_action(discard_decision)

        return discard_decision

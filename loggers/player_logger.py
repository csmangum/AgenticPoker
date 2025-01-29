import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class PlayerLogger:
    """Handles all logging operations for player-related actions."""

    @staticmethod
    def log_player_creation(name: str, starting_chips: int) -> None:
        """Log when a new player is created."""
        logger.info(f"New player created: {name} with ${starting_chips} chips")

    @staticmethod
    def log_bet_placement(
        player_name: str,
        amount: int,
        total_bet: int,
        chips_remaining: int,
        pot: Optional[int] = None,
    ) -> None:
        """Log when a player places a bet."""
        logger.debug(
            f"{player_name} bets ${amount} "
            f"(total bet: ${total_bet}, chips remaining: ${chips_remaining})"
        )
        if pot is not None:
            logger.debug(f"Pot after bet: ${pot}")

    @staticmethod
    def log_invalid_bet(player_name: str, reason: str) -> None:
        """Log when an invalid bet is attempted."""
        logger.warning(f"Invalid bet by {player_name}: {reason}")

    @staticmethod
    def log_state_reset(
        player_name: str, previous_bet: Optional[int] = None, context: str = "new round"
    ) -> None:
        """Log when a player's state is reset."""
        if previous_bet is not None:
            logger.debug(f"{player_name}'s bet reset from ${previous_bet} to $0")
        logger.debug(f"{player_name}'s state reset for {context}")

    @staticmethod
    def log_position_change(
        player_name: str, new_position: str, old_position: Optional[str] = None
    ) -> None:
        """Log when a player's position changes."""
        if old_position:
            logger.debug(
                f"{player_name}'s position changed from {old_position} to {new_position}"
            )
        else:
            logger.debug(f"{player_name} assigned position: {new_position}")

    @staticmethod
    def log_all_in(player_name: str, amount: int) -> None:
        """Log when a player goes all-in."""
        logger.info(f"{player_name} is all-in with ${amount}")

    @staticmethod
    def log_hand_dealt(player_name: str, hand: str, hidden: bool = True) -> None:
        """Log when a player is dealt cards."""
        if hidden:
            logger.debug(f"{player_name} dealt cards")
        else:
            logger.debug(f"{player_name} dealt: {hand}")

    @staticmethod
    def log_validation_error(error: str, value: Union[str, int]) -> None:
        """Log player validation errors."""
        logger.error(f"Player validation error: {error} (value: {value})")

    @staticmethod
    def log_chips_update(
        player_name: str, old_amount: int, new_amount: int, reason: str
    ) -> None:
        """Log when a player's chip count changes."""
        logger.debug(
            f"{player_name}'s chips: ${old_amount} -> ${new_amount} ({reason})"
        )

    @staticmethod
    def log_action_execution(
        player_name: str, action: str, amount: Optional[int] = None
    ) -> None:
        """Log when a player executes an action."""
        action_str = f"{action} ${amount}" if amount is not None else action
        logger.info(f"{player_name} executes: {action_str}")

    @staticmethod
    def log_action_error(player_name: str, action: str, error: Exception) -> None:
        """Log errors during action execution."""
        logger.error(f"Error executing {action} for {player_name}: {str(error)}")

    @staticmethod
    def log_prompt(player_name: str, prompt: str) -> None:
        """Log the prompt sent to the LLM."""
        logger.info(f"{player_name} prompt: {prompt}")

    @staticmethod
    def log_response(player_name: str, response: str) -> None:
        """Log the response received from the LLM."""
        logger.info(f"{player_name} response: {response}")

    @staticmethod
    def log_parsed_action(player_name: str, action: str) -> None:
        """Log the parsed action from the LLM response."""
        logger.info(f"{player_name} parsed action: {action}")

    @staticmethod
    def log_strategy(player_name: str, strategy: str) -> None:
        """Log the strategy planning thoughts."""
        logger.info(f"{player_name} strategy: {strategy}")

    @staticmethod
    def log_opponent_analysis(player_name: str, analysis: str) -> None:
        """Log the opponent analysis reflections."""
        logger.info(f"{player_name} opponent analysis: {analysis}")

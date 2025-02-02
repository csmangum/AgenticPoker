import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GameLogger:
    """Handles all logging operations for the main game flow and state changes."""

    @staticmethod
    def log_game_config(
        players: List[str],
        starting_chips: int,
        small_blind: int,
        big_blind: int,
        ante: int,
        max_rounds: Optional[int],
        session_id: Optional[str],
    ) -> None:
        """Log the initial game configuration."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Game Configuration")
        logger.info(f"{'='*50}")
        logger.info(f"Players: {', '.join(players)}")
        logger.info(f"Starting chips: ${starting_chips}")
        logger.info(f"Blinds: ${small_blind}/${big_blind}")
        logger.info(f"Ante: ${ante}")
        if max_rounds:
            logger.info(f"Max rounds: {max_rounds}")
        if session_id:
            logger.info(f"Session ID: {session_id}")
        logger.info(f"{'='*50}\n")

    @staticmethod
    def log_round_header(round_number: int) -> None:
        """Log the start of a new round."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Round {round_number}")
        logger.info(f"{'='*50}")

    @staticmethod
    def log_chip_counts(
        chips_dict: Dict[str, int],
        message: str,
        show_short_stack: bool = False,
        big_blind: Optional[int] = None,
    ) -> None:
        """Log chip counts for all players."""
        logger.info(f"\n{message}:")
        for player_name, chips in sorted(
            chips_dict.items(), key=lambda x: x[1], reverse=True
        ):
            chips_str = f"${chips}"
            if show_short_stack and big_blind and chips < big_blind:
                chips_str += " (short stack)"
            logger.info(f"  {player_name}: {chips_str}")

    @staticmethod
    def log_table_positions(positions: Dict[str, str]) -> None:
        """Log the current table positions."""
        logger.info("\nTable positions:")
        for position, player_name in positions.items():
            logger.info(f"  {position}: {player_name}")

    @staticmethod
    def log_betting_structure(
        small_blind: int, big_blind: int, ante: int, min_bet: int
    ) -> None:
        """Log the current betting structure."""
        logger.info("\nBetting structure:")
        logger.info(f"  Small blind: ${small_blind}")
        logger.info(f"  Big blind: ${big_blind}")
        if ante > 0:
            logger.info(f"  Ante: ${ante}")
        logger.info(f"  Minimum bet: ${min_bet}")

    @staticmethod
    def log_phase_header(phase: str) -> None:
        """Log the start of a game phase."""
        logger.info(f"\n====== {phase} ======\n")

    @staticmethod
    def log_phase_complete(phase: str) -> None:
        """Log the completion of a game phase."""
        logger.info(f"====== {phase} Complete ======\n")

    @staticmethod
    def log_player_elimination(player_name: str) -> None:
        """Log when a player is eliminated."""
        logger.info(f"\n{player_name} is eliminated (out of chips)!")

    @staticmethod
    def log_game_winner(winner_name: str, chips: int) -> None:
        """Log the game winner."""
        logger.info(f"\nGame Over! {winner_name} wins with ${chips}!")

    @staticmethod
    def log_all_bankrupt() -> None:
        """Log when all players are bankrupt."""
        logger.info("\nGame Over! All players are bankrupt!")

    @staticmethod
    def log_max_rounds_reached(rounds: int) -> None:
        """Log when maximum rounds is reached."""
        logger.info(f"\nGame ended after {rounds} rounds!")

    @staticmethod
    def log_game_summary(
        rounds_played: int,
        max_rounds: Optional[int],
        final_standings: List[Dict[str, any]],
    ) -> None:
        """Log the final game summary."""
        logger.info("\n=== Game Summary ===")
        logger.info(f"Total rounds played: {rounds_played}")
        if max_rounds and rounds_played >= max_rounds:
            logger.info("Game ended due to maximum rounds limit")

        logger.info("\nFinal Standings:")
        for i, player in enumerate(final_standings, 1):
            status = " (eliminated)" if player["eliminated"] else ""
            logger.info(f"{i}. {player['name']}: ${player['chips']}{status}")

    @staticmethod
    def log_deck_status(remaining_cards: int, context: str = "") -> None:
        """Log the current deck status."""
        context_str = f" for {context}" if context else ""
        logger.info(f"Cards remaining{context_str}: {remaining_cards}")

    @staticmethod
    def log_game_ended_after_rounds(max_rounds: int) -> None:
        """Log the game ended after a certain number of rounds."""
        logger.info(f"\nGame ended after {max_rounds} rounds!")

    @staticmethod
    def log_new_deck_shuffled(round_number: int) -> None:
        """Log the new deck shuffled for a round."""
        logger.info(f"New deck shuffled for round {round_number}")

    @staticmethod
    def log_skip_betting(reason: str) -> None:
        """Log when betting round is skipped.
        
        Args:
            reason (str): The reason why betting was skipped
        """
        logger.info(f"Skipping betting round: {reason}")

    @staticmethod
    def log_info(message: str) -> None:
        """Log an informational message."""
        logger.info(message)

    @staticmethod
    def log_warning(message: str) -> None:
        """Log a warning message."""
        logger.warning(message)

    @staticmethod
    def log_error(message: str) -> None:
        """Log an error message."""
        logger.error(message)

from typing import Dict, List, Optional
import os
import shutil
import time

from data.db_client import DatabaseClient
from data.enums import ActionType
from data.states.game_state import GameState
from data.states.round_state import RoundState
from data.types.action_decision import ActionDecision
from game.config import GameConfig
from game.table import Table
from loggers.game_logger import GameLogger
from loggers.table_logger import TableLogger

from . import betting, draw, showdown
from .deck import Deck
from .hand import Hand
from .player import Player
from .pot import Pot

#! create pre and post helper methods for standard setup and teardown (might already have this)
#! db game state at the start of the game
#! db round state at the start of each round
#! db agent state after each action (i.e. R1A1, R1A2, R2A1, R2A2, etc.)
#! futrure db for every agent thoughts which are prompt: response pairs


class AgenticPoker:
    """
    A comprehensive 5-card draw poker game manager that handles game flow and player interactions.

    This class manages the complete lifecycle of a poker game, including:
    - Player management and chip tracking
    - Dealing cards and managing the deck
    - Betting rounds and pot management
    - Side pot creation and resolution
    - Winner determination and chip distribution
    - Game state tracking and logging

    The game follows standard 5-card draw poker rules with configurable betting structures
    including blinds, antes, and various betting limits.

    Attributes:
        deck (Deck): The deck of cards used for dealing
        table (Table): The table of players in the game
        small_blind (int): Required small blind bet amount
        big_blind (int): Required big blind bet amount
        dealer_index (int): Position of current dealer (0-based, moves clockwise)
        round_number (int): Current round number (increments at start of each round)
        max_rounds (Optional[int]): Maximum number of rounds to play, or None for unlimited
        ante (int): Mandatory bet required from all players at start of each hand
        session_id (Optional[str]): Unique identifier for this game session
        round_starting_stacks (Dict[Player, int]): Dictionary of starting chip counts for each round
        config (GameConfig): Configuration parameters for the game
        current_bet (int): Current bet amount that players must match
        pot (Pot): Manages pot calculations and side pot creation
        logger (Logger): Logger instance for game events and state changes

    Example:
        >>> players = ["Alice", "Bob", "Charlie"]
        >>> game = AgenticPoker(players, small_blind=10, big_blind=20)
        >>> game.start_game()
    """

    deck: Deck
    table: Table
    small_blind: int
    big_blind: int
    dealer_index: int
    round_number: int
    max_rounds: Optional[int]
    ante: int
    session_id: Optional[str]
    round_starting_stacks: Dict[Player, int]
    config: GameConfig
    current_bet: int
    pot: Pot
    last_raiser: Optional[Player]
    db_client: DatabaseClient

    def __init__(
        self,
        players: List[Player],
        small_blind: int = 10,
        big_blind: int = 20,
        ante: int = 0,
        session_id: Optional[str] = None,
        config: Optional[GameConfig] = None,
    ) -> None:
        """Initialize a new poker game with specified players and configuration."""

        if not players:
            raise ValueError("Must provide at least 2 players")

        # Clean up any existing ChromaDB databases in results directory
        self._cleanup_chroma_dbs()

        # Support both direct parameter initialization and GameConfig
        if config:
            self.config = config
        else:
            self.config = GameConfig(
                small_blind=small_blind,
                big_blind=big_blind,
                ante=ante,
                session_id=session_id,
            )

        self.session_id = self.config.session_id
        self.deck = Deck()

        # Convert names to players if needed
        self.table = Table(players)
        # Validate player chips
        if any(p.chips < 0 for p in self.table):
            raise ValueError("Players cannot have negative chips")

        #! make into betting class
        self.current_bet = 0
        self.pot = Pot()

        self.small_blind = self.config.small_blind
        self.big_blind = self.config.big_blind
        self.dealer_index = 0
        self.round_number = 0  # Initialize only round_number
        self.max_rounds = self.config.max_rounds
        self.ante = self.config.ante
        self.last_raiser = None
        self.initial_chips = {}

        # Replace logging with GameLogger
        GameLogger.log_game_config(
            players=[p.name for p in self.table],
            starting_chips=self.config.starting_chips,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            ante=self.config.ante,
            max_rounds=self.config.max_rounds,
            session_id=self.config.session_id,
        )

        # Initialize database client
        self.db_client = DatabaseClient()

    def _cleanup_chroma_dbs(self) -> None:
        """Clean up any existing ChromaDB databases in the results directory."""
        results_dir = os.path.join(os.getcwd(), "results")
        chroma_dir = os.path.join(results_dir, "chroma_db")
        
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                time.sleep(0.2)  # Give OS time to release handles
                GameLogger.log_info("Cleared previous ChromaDB databases")
            except PermissionError:
                # Wait a bit longer and try again
                time.sleep(1.0)
                try:
                    shutil.rmtree(chroma_dir)
                    GameLogger.log_info("Cleared previous ChromaDB databases after retry")
                except PermissionError:
                    GameLogger.log_info(
                        "Could not clear ChromaDB directory - will create new collections"
                    )

    def play_game(self, max_rounds: Optional[int] = None) -> None:
        """
        Execute the main game loop until a winner is determined.

        This method manages the core game flow through multiple rounds until completion.
        Each round consists of four phases:
        1. Pre-draw betting
        2. Draw (card exchange)
        3. Post-draw betting
        4. Showdown (if multiple players remain)

        The game continues until one of these conditions is met:
        - Only one player remains with chips
        - Maximum number of rounds is reached (if configured)
        - All players are eliminated (bankrupt)

        Side Effects:
            - Updates player chip counts
            - Tracks eliminated players
            - Logs game progress and results
            - Rotates dealer position between rounds
            - Updates game state (round number, pot, etc.)

        Example round flow:
            1. Check for eliminations and game end conditions
            2. Start new round (deal cards, collect blinds/antes)
            3. Execute betting/drawing phases
            4. Distribute pot to winner(s)
            5. Reset for next round
        """
        eliminated_players = []

        # Get game state and convert to dict before saving
        game_state = self.get_state()
        self.db_client.save_game_snapshot(
            self.session_id, self.round_number, game_state
        )

        if max_rounds:
            self.max_rounds = max_rounds

        while len(self.table) > 1:
            self.round_number += 1

            # Check max rounds before starting new round
            if self.max_rounds and self.round_number > self.max_rounds:
                GameLogger.log_game_ended_after_rounds(self.max_rounds)
                break

            # Handle eliminations and check if game should end
            if not self._handle_player_eliminations(eliminated_players):
                break

            self._start_new_round()

            should_continue = self._handle_pre_draw_phase()

            if should_continue:
                self._handle_draw_phase()

            if should_continue:
                should_continue = self._handle_post_draw_phase()

            self._handle_showdown()

            self._reset_round()
            # Save round state after converting to dict
            self.db_client.save_round_snapshot(
                self.session_id, self.round_number, self.round_state
            )

        self._log_game_summary(eliminated_players)

        # Ensure database session is cleaned up
        self.db_client.close()

    def _handle_pre_draw_phase(self) -> bool:
        GameLogger.log_phase_header("Pre-draw betting")

        # Record pre-draw state for agents
        for player in self.table:
            if hasattr(player, "record_game_state"):
                player.current_round_id = f"round_{self.round_number}_predraw"
                player.record_game_state(self)

        should_continue = betting.handle_betting_round(self)

        # Calculate side pots if any player is all-in
        if any(p.is_all_in for p in self.table.players):
            self.pot.calculate_side_pots(self.table.players)

        # Move bets to pot and reset them
        self.pot.end_betting_round(self.table.players)

        GameLogger.log_phase_complete("Pre-draw betting")
        return should_continue

    def _handle_draw_phase(self) -> bool:
        GameLogger.log_phase_header("Draw Phase")

        # Record draw phase state for agents
        for player in self.table:
            if hasattr(player, "record_game_state"):
                player.current_round_id = f"round_{self.round_number}_draw"
                player.record_game_state(self)

        should_continue = draw.handle_draw_phase(self)
        GameLogger.log_phase_complete("Draw Phase")
        return should_continue

    def _handle_post_draw_phase(self) -> bool:
        """Handle the post-draw betting phase.

        The order of operations is important:
        1. Run the betting round
        2. Calculate side pots while bets are still set
        3. End betting round which moves bets to pot
        """
        GameLogger.log_phase_header("Post-draw betting")

        # Record post-draw state for agents
        for player in self.table:
            if hasattr(player, "record_game_state"):
                player.current_round_id = f"round_{self.round_number}_postdraw"
                player.record_game_state(self)

        # Skip post-draw betting if everyone is all-in
        if all(p.is_all_in or p.folded for p in self.table.players):
            GameLogger.log_skip_betting("All remaining players are all-in")
            return True

        # First handle the betting
        should_continue = betting.handle_betting_round(self)

        # Calculate side pots if any player is all-in
        if any(p.is_all_in for p in self.table.players):
            self.pot.calculate_side_pots(self.table.players)

        # Move bets to pot and reset them
        self.pot.end_betting_round(self.table.players)

        GameLogger.log_phase_complete("Post-draw betting")
        return should_continue

    def _handle_showdown(self) -> None:
        GameLogger.log_phase_header("Showdown")

        # Record showdown results for agents
        for player in self.table:
            if hasattr(player, "record_hand_result"):
                # Determine if player won and their profit/loss
                winners = showdown._evaluate_hands(
                    [p for p in self.table.players if not p.folded]
                )
                won = player in winners
                chips_delta = player.chips - self.initial_chips.get(player, 0)

                # Get opponent cards if they showed down
                opponent_cards = {}
                for opp in self.table:
                    if opp != player and not opp.folded:
                        opponent_cards[opp.name] = str(opp.hand)

                player.record_hand_result(
                    won=won,
                    amount=chips_delta,
                    opponent_cards=str(opponent_cards) if opponent_cards else "",
                )

        showdown.handle_showdown(
            players=self.table.players,
            initial_chips=self.initial_chips,
            pot=self.pot,
        )
        GameLogger.log_phase_complete("Showdown")

    def _start_new_round(self) -> None:
        """
        Start a new round of poker by initializing the round state and collecting mandatory bets.

        This method:
        1. Initializes the round state (new deck, deal cards, reset bets)
        2. Logs the round information and current game state
        3. Collects blinds and antes from players
        4. Processes any pre-round AI player messages

        Side Effects:
            - Deals new cards to players
            - Collects blinds and antes
            - Updates pot and player chip counts
            - Logs round information
            - Processes AI player messages
        """
        # Start new round with remaining players
        #! maybe have eligible_players as a property of player queue
        eligible_players = [p for p in self.table if p.chips > 0]

        # Store initial chips before starting round
        self.initial_chips = {p: p.chips for p in eligible_players}

        # Remove players with 0 chips
        for player in self.table:
            if player.chips == 0:
                self.table.remove_player(player)

        self._initialize_round()
        self._log_round_info()

        # Record new round state for each agent
        for player in self.table:
            if hasattr(player, "record_game_state"):
                player.current_hand_id = f"hand_{self.round_number}"
                player.current_round_id = f"round_{self.round_number}"
                player.record_game_state(self)

        self._collect_blinds_and_antes()

        # Add this check: If all players except one are all-in after blinds/antes,
        # skip straight to showdown
        active_not_allin = [p for p in self.table if not p.is_all_in and not p.folded]
        if len(active_not_allin) <= 1:
            return

        # Handle AI player pre-round messages
        #! need to fix this
        # for player in self.table.players:
        #     if hasattr(player, "get_message"):
        #         game_state = f"Round {self.round_number}, Your chips: ${player.chips}"
        #         message = player.get_message(game_state)

    def _collect_blinds_and_antes(self) -> None:
        """
        Collect mandatory bets (blinds and antes) at the start of each hand.

        This method:
        1. Stores the starting chip stacks for each player
        2. Collects small blind, big blind, and antes from appropriate players
        3. Sets the current bet to the big blind amount
        4. Updates the pot with all collected chips

        Side Effects:
            - Updates self.round_starting_stacks with initial chip counts
            - Updates players' chip counts as they post blinds/antes
            - Sets self.current_bet to the big blind amount
            - Updates self.pot with collected chips

        Note:
            - Small blind is posted by player to left of dealer
            - Big blind is posted by player to left of small blind
            - Antes are collected from all players if configured
        """
        # Store starting stacks
        self.round_starting_stacks = {p: p.chips for p in self.table}

        # Use betting module to collect blinds and antes
        collected = betting.collect_blinds_and_antes(
            dealer_index=self.dealer_index,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            ante=self.ante,
            game=self,
        )

        # Set the current bet to the big blind amount
        self.current_bet = self.big_blind

        # Update pot directly with collected amount
        self.pot.pot = collected

        # Record blind/ante actions for agents
        for player in self.table:
            if hasattr(player, "record_action"):
                if player.bet == self.small_blind:
                    player.record_action(
                        ActionDecision(
                            action_type=ActionType.SMALL_BLIND,
                            raise_amount=self.small_blind,
                        ),
                        self,
                    )
                elif player.bet == self.big_blind:
                    player.record_action(
                        ActionDecision(
                            action_type=ActionType.BIG_BLIND,
                            raise_amount=self.big_blind,
                        ),
                        self,
                    )

    def _log_round_info(self) -> None:
        """Log complete round state including stacks, positions, and betting information."""
        # Log round header
        GameLogger.log_round_header(self.round_number)

        # Log chip stacks
        GameLogger.log_chip_counts(
            {p.name: self.round_starting_stacks[p] for p in self.table},
            "Starting stacks (before antes/blinds)",
            show_short_stack=True,
            big_blind=self.big_blind,
        )

        # Log table positions
        positions = {}
        players_count = len(self.table)
        for i in range(players_count):
            position_index = (self.dealer_index + i) % players_count
            player = self.table[position_index]
            position = (
                "Dealer"
                if i == 0
                else (
                    "Small Blind"
                    if i == 1
                    else "Big Blind" if i == 2 else f"Position {i}"
                )
            )
            positions[position] = player.name
        GameLogger.log_table_positions(positions)

        # Log betting structure
        GameLogger.log_betting_structure(
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            ante=self.ante,
            min_bet=self.config.min_bet,
        )

    def _handle_player_eliminations(self, eliminated_players: List[Player]) -> bool:
        # Track newly eliminated players
        for player in self.table:
            if player.chips <= 0 and player not in eliminated_players:
                eliminated_players.append(player)
                GameLogger.log_player_elimination(player.name)
                # Remove eliminated player from table immediately
                self.table.remove_player(player)

        # Check game end conditions - do this after removing eliminated players
        if len(self.table) <= 1:
            if len(self.table) == 1:
                GameLogger.log_game_winner(self.table[0].name, self.table[0].chips)
            else:
                GameLogger.log_all_bankrupt()
            return False

        return True

    def _log_game_summary(self, eliminated_players: List[Player]) -> None:
        """Log a summary of the game results."""
        # Get all players (both active and eliminated)
        all_players = list(
            {player for player in (self.table.players + eliminated_players)}
        )

        # Sort players by final chip count
        all_players.sort(key=lambda p: p.chips, reverse=True)

        # Log final standings
        GameLogger.log_game_summary(
            rounds_played=self.round_number,
            max_rounds=self.max_rounds,
            final_standings=[
                {
                    "name": player.name,
                    "chips": player.chips,
                    "eliminated": player in eliminated_players,
                }
                for player in all_players
            ],
        )

    def _initialize_round(self) -> None:
        """Initialize the state for a new round of poker."""
        # Reset round state
        self.pot.reset_pot()

        # Store initial chips BEFORE any deductions
        self.round_starting_stacks = {p: p.chips for p in self.table}

        # Reset player states
        for player in self.table:
            player.bet = 0
            player.folded = False

        # Create and shuffle a fresh deck for the new round
        self.deck = Deck()
        self.deck.shuffle()
        GameLogger.log_new_deck_shuffled(self.round_number)

        # Deal initial hands
        self._deal_cards()

        # Log deck status after initial deal
        GameLogger.log_deck_status(self.deck.remaining_cards(), "after initial deal")

        # Create new round state
        self.round_state = RoundState.new_round(self.round_number)

    def _reset_round(self) -> None:
        """Reset the state after a round is complete."""
        # Clear hands and bets
        for player in self.table:
            player.reset_for_new_round()

        # Reset pot in pot
        self.pot.reset_pot()

        # Rotate dealer position for next round
        self.dealer_index = (self.dealer_index + 1) % len(self.table)

    def get_state(self) -> GameState:
        return GameState.from_game(self)

    def _deal_cards(self) -> None:
        """Deal new hands to all players."""
        for player in self.table:
            player.bet = 0
            player.folded = False
            player.hand = Hand()
            player.hand.add_cards(self.deck.deal(5))

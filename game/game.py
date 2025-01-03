import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from . import betting, draw, post_draw, pre_draw
from .deck import Deck
from .hand import Hand
from .player import Player
from .pot_manager import PotManager
from .utils import log_chip_movements


@dataclass
class GameConfig:
    """
    Configuration parameters for a poker game.

    This class defines all the customizable parameters that control game behavior,
    including betting limits, starting conditions, and round restrictions.

    Attributes:
        starting_chips (int): Initial chip amount for each player (default: 1000)
        small_blind (int): Small blind bet amount (default: 10)
        big_blind (int): Big blind bet amount (default: 20)
        ante (int): Mandatory bet required from all players before dealing (default: 0)
        max_rounds (Optional[int]): Maximum number of rounds to play, None for unlimited (default: None)
        session_id (Optional[str]): Unique identifier for the game session (default: None)
        max_raise_multiplier (int): Maximum raise as multiplier of current bet (default: 3)
        max_raises_per_round (int): Maximum number of raises allowed per betting round (default: 4)
        min_bet (Optional[int]): Minimum bet amount, defaults to big blind if not specified

    Raises:
        ValueError: If any of the numerical parameters are invalid (negative or zero where not allowed)
    """

    starting_chips: int = 1000
    small_blind: int = 10
    big_blind: int = 20
    ante: int = 0
    max_rounds: Optional[int] = None
    session_id: Optional[str] = None
    max_raise_multiplier: int = 3
    max_raises_per_round: int = 4
    min_bet: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.starting_chips <= 0:
            raise ValueError("Starting chips must be positive")
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.ante < 0:
            raise ValueError("Ante cannot be negative")
        if self.max_raise_multiplier <= 0:
            raise ValueError("Max raise multiplier must be positive")
        if self.max_raises_per_round <= 0:
            raise ValueError("Max raises per round must be positive")
        # Set min_bet to big blind if not specified
        if self.min_bet is None:
            self.min_bet = self.big_blind
        elif self.min_bet < self.big_blind:
            raise ValueError("Minimum bet cannot be less than big blind")


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
        players (List[Player]): List of currently active players in the game
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
        pot_manager (PotManager): Manages pot calculations and side pot creation
        logger (Logger): Logger instance for game events and state changes

    Example:
        >>> players = ["Alice", "Bob", "Charlie"]
        >>> game = AgenticPoker(players, small_blind=10, big_blind=20)
        >>> game.start_game()
    """

    # Class attributes defined before __init__
    deck: Deck
    players: List[Player]
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

    def __init__(
        self,
        players: Union[List[str], List[Player]],
        small_blind: int = 10,
        big_blind: int = 20,
        ante: int = 0,
        session_id: Optional[str] = None,
        config: Optional[GameConfig] = None,
    ) -> None:
        """Initialize a new poker game with specified players and configuration."""
        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        if not players:
            raise ValueError("Must provide at least 2 players")

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
        if players and isinstance(players[0], str):
            self.players = [
                Player(name, self.config.starting_chips) for name in players
            ]
        else:
            self.players = players  # Use provided Player objects
            # Validate player chips
            if any(p.chips < 0 for p in self.players):
                raise ValueError("Players cannot have negative chips")

        self.current_bet = 0  # Add this line to initialize current_bet
        self.pot_manager = PotManager()

        self.small_blind = self.config.small_blind
        self.big_blind = self.config.big_blind
        self.dealer_index = 0
        self.round_number = 0  # Initialize only round_number
        self.max_rounds = self.config.max_rounds
        self.ante = self.config.ante

        # Log game configuration
        logging.info(f"\n{'='*50}")
        logging.info(f"Game Configuration")
        logging.info(f"{'='*50}")
        logging.info(f"Players: {', '.join([p.name for p in self.players])}")
        logging.info(f"Starting chips: ${self.config.starting_chips}")
        logging.info(f"Blinds: ${self.config.small_blind}/${self.config.big_blind}")
        logging.info(f"Ante: ${self.config.ante}")
        if self.config.max_rounds:
            logging.info(f"Max rounds: {self.config.max_rounds}")
        if self.config.session_id:
            logging.info(f"Session ID: {self.config.session_id}")
        logging.info(f"{'='*50}\n")

    def start_game(self) -> None:
        """
        Execute the main game loop until a winner is determined or max rounds reached.
        """
        eliminated_players = []

        while len(self.players) > 1:
            self.round_number += 1

            # Check max rounds before starting new round
            if self.max_rounds and self.round_number > self.max_rounds:
                logging.info(f"\nGame ended after {self.max_rounds} rounds!")
                break

            # Handle eliminations and check if game should end
            if not self._handle_player_eliminations(eliminated_players):
                break

            # Start new round with remaining players
            self.players = [p for p in self.players if p.chips > 0]

            # Store initial chips before starting round
            initial_chips = {p: p.chips for p in self.players}

            self.start_round()

            # Pre-draw betting round
            game_state = self._create_game_state()
            new_pot, side_pots, should_continue = pre_draw.handle_pre_draw_betting(
                players=self.players,
                pot=self.pot_manager.pot,
                dealer_index=self.dealer_index,
                game_state=game_state,
            )

            # Update pot manager with new pot and side pots
            if side_pots:
                self.pot_manager.side_pots = side_pots
            self.pot_manager.pot = new_pot

            if not should_continue:
                self._reset_round()
                continue

            # Draw phase
            draw.handle_draw_phase(players=self.players, deck=self.deck)

            # Post-draw betting round
            game_state = self._create_game_state()
            new_pot, side_pots, should_continue = post_draw.handle_post_draw_betting(
                players=self.players,
                pot=self.pot_manager.pot,
                dealer_index=self.dealer_index,
                game_state=game_state,
            )

            # Update pot manager with new pot and side pots from post-draw betting
            if side_pots:
                self.pot_manager.side_pots = side_pots
            self.pot_manager.pot = new_pot

            if not should_continue:
                self._reset_round()
                continue

            # Showdown
            post_draw.handle_showdown(
                players=self.players,
                initial_chips=initial_chips,
                pot_manager=self.pot_manager,
            )

            self._reset_round()

        self._log_game_summary(eliminated_players)

    def start_round(self) -> None:
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
        self._initialize_round()

        # Log round info BEFORE collecting antes/blinds
        self._log_round_info()

        # Collect blinds and antes AFTER logging initial state
        self._collect_blinds_and_antes()

        # Handle AI player pre-round messages
        for player in self.players:
            if hasattr(player, "get_message"):
                game_state = f"Round {self.round_number}, Your chips: ${player.chips}"
                message = player.get_message(game_state)

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
            - Updates self.pot_manager with collected chips

        Note:
            - Small blind is posted by player to left of dealer
            - Big blind is posted by player to left of small blind
            - Antes are collected from all players if configured
        """
        # Store starting stacks
        self.round_starting_stacks = {p: p.chips for p in self.players}

        # Use betting module to collect blinds and antes
        collected = betting.collect_blinds_and_antes(
            players=self.players,
            dealer_index=self.dealer_index,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            ante=self.ante,
        )

        # Set the current bet to the big blind amount
        self.current_bet = self.big_blind

        # Update pot through pot manager
        self.pot_manager.add_to_pot(collected)

    def _log_chip_counts(
        self,
        chips_dict: Dict[Player, int],
        message: str,
        show_short_stack: bool = False,
    ) -> None:
        """
        Log chip counts for all players.

        Args:
            chips_dict: Dictionary mapping players to their chip counts
            message: Header message to display
            show_short_stack: Whether to show short stack warnings
        """
        logging.info(f"\n{message}:")
        sorted_players = sorted(self.players, key=lambda p: chips_dict[p], reverse=True)
        for player in sorted_players:
            chips_str = f"${chips_dict[player]}"
            if show_short_stack and chips_dict[player] < self.big_blind:
                chips_str += " (short stack)"
            logging.info(f"  {player.name}: {chips_str}")

    def _log_round_header(self) -> None:
        """Log the round number with separator lines."""
        logging.info(f"\n{'='*50}")
        logging.info(f"Round {self.round_number}")
        logging.info(f"{'='*50}")

    def _log_round_info(self) -> None:
        """
        Log complete round state including stacks, positions, and betting information.
        """
        # Log round header
        self._log_round_header()

        # Log chip stacks
        self._log_chip_counts(
            self.round_starting_stacks,
            "Starting stacks (before antes/blinds)",
            show_short_stack=True,
        )

        # Log table positions
        logging.info("\nTable positions:")
        players_count = len(self.players)
        for i in range(players_count):
            position_index = (self.dealer_index + i) % players_count
            player = self.players[position_index]
            position = ""
            if i == 0:
                position = "Dealer"
            elif i == 1:
                position = "Small Blind"
            elif i == 2:
                position = "Big Blind"
            else:
                position = f"Position {i}"
            logging.info(f"  {position}: {player.name}")

        # Log betting structure
        logging.info("\nBetting structure:")
        logging.info(f"  Small blind: ${self.small_blind}")
        logging.info(f"  Big blind: ${self.big_blind}")
        if self.ante > 0:
            logging.info(f"  Ante: ${self.ante}")
        logging.info(f"  Minimum bet: ${self.config.min_bet}")

        # Log side pots using pot manager's method
        if self.pot_manager.side_pots:
            self.pot_manager.log_side_pots(logging)

    def _handle_player_eliminations(self, eliminated_players: List[Player]) -> bool:
        """
        Handle player eliminations and determine if the game should continue.

        Args:
            eliminated_players: List to track eliminated players for game history

        Returns:
            bool: True if game can continue, False if game should end

        Side Effects:
            - Updates players list
            - Updates eliminated_players list
            - Logs elimination messages
        """
        # Track newly eliminated players first
        for player in self.players:
            if player.chips <= 0 and player not in eliminated_players:
                eliminated_players.append(player)
                logging.info(f"\n{player.name} is eliminated (out of chips)!")

        # Remove bankrupt players from active game
        self.players = [player for player in self.players if player.chips > 0]

        # Check game end conditions
        if len(self.players) == 1:
            logging.info(
                f"\nGame Over! {self.players[0].name} wins with ${self.players[0].chips}!"
            )
            return False
        elif len(self.players) == 0:
            logging.info("\nGame Over! All players are bankrupt!")
            return False

        return True

    def _create_game_state(self) -> dict:
        """Create a dictionary containing the current game state."""
        # Calculate big blind position (2 seats after dealer)
        big_blind_pos = (self.dealer_index + 2) % len(self.players)
        big_blind_player = self.players[big_blind_pos]

        return {
            "pot": self.pot_manager.pot,
            "players": [
                {
                    "name": p.name,
                    "chips": p.chips,
                    "bet": p.bet,
                    "folded": p.folded,
                    "position": i,
                }
                for i, p in enumerate(self.players)
            ],
            "current_bet": max(p.bet for p in self.players) if self.players else 0,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "dealer_index": self.dealer_index,
            "big_blind_position": big_blind_player,  # Add big blind player to game state
        }

    def _log_chip_movements(self, initial_chips: Dict[Player, int]) -> None:
        """Log the chip movements for each player from their initial amounts."""
        log_chip_movements(self.players, initial_chips)

    def _log_game_summary(self, eliminated_players: List[Player]) -> None:
        """Log the final game summary and standings."""
        logging.info("\n=== Game Summary ===")
        logging.info(f"Total rounds played: {self.round_number}")
        if self.max_rounds and self.round_number >= self.max_rounds:
            logging.info("Game ended due to maximum rounds limit")

        # Use a set to ensure unique players
        all_players = list({player for player in (self.players + eliminated_players)})
        # Sort by chips (eliminated players will have 0)
        all_players.sort(key=lambda p: p.chips, reverse=True)

        logging.info("\nFinal Standings:")
        for i, player in enumerate(all_players, 1):
            status = " (eliminated)" if player in eliminated_players else ""
            logging.info(f"{i}. {player.name}: ${player.chips}{status}")

    def _initialize_round(self) -> None:
        """Initialize the state for a new round of poker."""
        # Reset round state
        self.pot_manager.reset_pot()

        # Store initial chips BEFORE any deductions
        self.round_starting_stacks = {p: p.chips for p in self.players}

        # Reset player states
        for player in self.players:
            player.bet = 0
            player.folded = False

        # Create and shuffle a fresh deck for the new round
        self.deck = Deck()
        self.deck.shuffle()
        logging.info(f"New deck shuffled for round {self.round_number}")

        # Deal initial hands
        self._deal_cards()

        # Log deck status after initial deal
        logging.info(
            f"Cards remaining after initial deal: {self.deck.remaining_cards()}"
        )

    def _reset_round(self) -> None:
        """Reset the state after a round is complete."""
        # Clear hands and bets
        for player in self.players:
            player.bet = 0
            player.folded = False
            if hasattr(player, "hand"):
                player.hand = None

        # Reset pot in pot_manager
        self.pot_manager.reset_pot()

        # Rotate dealer position for next round
        self.dealer_index = (self.dealer_index + 1) % len(self.players)

    def _deal_cards(self) -> None:
        """Deal new hands to all players."""
        for player in self.players:
            player.bet = 0
            player.folded = False
            player.hand = Hand()
            player.hand.add_cards(self.deck.deal(5))

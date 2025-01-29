import logging

from data.states.player_state import PlayerState
from data.types.action_decision import ActionDecision, ActionType
from data.types.player_types import PlayerPosition
from loggers.player_logger import PlayerLogger

from .hand import Hand


class Player:
    """
    Represents a poker player with chips, betting capability, and a hand of cards.

    This class manages a player's state during a poker game, including their chips,
    current bet, hand of cards, and fold status. It handles all betting operations
    and ensures proper tracking of player resources.

    Attributes:
        name (str): The player's display name
        chips (int): The current amount of chips the player possesses
        bet (int): The amount of chips bet in the current betting round
        folded (bool): Whether the player has folded in the current hand
        hand (Hand): The player's current hand of cards
        position (PlayerPosition): The player's current position in the game
    """

    name: str
    chips: int  #! change to stack???
    bet: int
    folded: bool
    hand: Hand
    position: PlayerPosition
    is_all_in: bool  #! change to all_in
    checked: bool
    called: bool

    def __init__(self, name: str, chips: int = 1000) -> None:
        """
        Initialize a new player with a name and starting chips.

        Args:
            name (str): The player's display name
            chips (int, optional): Starting amount of chips. Defaults to 1000.

        Raises:
            ValueError: If name is empty or chips is negative
        """
        if not name or name.isspace():
            raise ValueError("Player name cannot be empty or whitespace")
        if not isinstance(chips, int):
            raise ValueError("Chips must be an integer value")
        if chips < 0:
            raise ValueError("Cannot initialize player with negative chips")

        self.name = name
        self.chips = chips
        self.bet = 0
        self.hand = Hand()
        self.position = PlayerPosition.OTHER
        self.folded = False
        self.is_all_in = False
        self.checked = False
        self.called = False

        PlayerLogger.log_player_creation(name, chips)

    def place_bet(self, amount: int, game) -> int:
        """Place a bet, ensuring it doesn't exceed available chips."""
        if amount < 0:
            PlayerLogger.log_invalid_bet(self.name, "Cannot place negative bet")
            raise ValueError("Cannot place negative bet")

        # If player doesn't have enough chips, they go all-in
        if amount >= self.chips:
            amount = self.chips
            self.is_all_in = True
            PlayerLogger.log_all_in(self.name, amount)

        self.chips -= amount
        self.bet += amount
        game.pot.add_to_pot(amount)

        PlayerLogger.log_bet_placement(
            self.name, amount, self.bet, self.chips, game.pot.pot
        )
        return amount

    def execute(self, action_decision: ActionDecision, game):
        try:
            PlayerLogger.log_action_execution(
                self.name, action_decision.action_type.name
            )
            if action_decision.action_type == ActionType.RAISE:
                self._raise(action_decision.raise_amount, game)
            elif action_decision.action_type == ActionType.CALL:
                self._call(game.current_bet, game)
            elif action_decision.action_type == ActionType.CHECK:
                self._check()
            elif action_decision.action_type == ActionType.FOLD:
                self._fold()
        except Exception as e:
            PlayerLogger.log_action_error(
                self.name, action_decision.action_type.name, e
            )
            raise

    def _raise(self, amount: int, game) -> None:
        """Handle raise action with proper all-in logic."""
        # Get current raise count and minimum bet
        raise_count = game.round_state.raise_count if game.round_state else 0
        min_bet = game.config.min_bet

        # Check if we've hit max raises
        if raise_count >= game.config.max_raises_per_round:
            self._call(game.current_bet, game)
            return

        # If player can't make minimum raise, they must go all-in or fold
        if self.chips < (game.current_bet + min_bet):
            if self.chips > game.current_bet:  # Can at least call
                self._call(self.chips, game)  # All-in call
            else:
                self._fold()
            return

        # Normal raise handling
        actual_amount = min(amount, self.chips)
        self.place_bet(actual_amount, game)

        if actual_amount > game.current_bet:
            game.last_raiser = self
            if game.round_state is not None:
                game.round_state.raise_count += 1
                game.round_state.last_raiser = self.name

    def _call(self, amount: int, game) -> None:
        """
        Mark the player as called for the current hand.
        If player cannot meet the call amount, they are forced to fold.
        """
        # Force fold if player cannot meet the call amount
        if amount > self.chips:
            self._fold()
            return

        self.place_bet(amount, game)
        self.called = True

        if self.chips == 0:
            self.is_all_in = True
            PlayerLogger.log_all_in(self.name, amount)

    def _check(self) -> None:
        """
        Mark the player as checked for the current hand.
        """
        self.checked = True

    def _fold(self) -> None:
        """
        Mark the player as folded for the current hand.

        Side Effects:
            - Sets player's folded status to True
            - Logs the fold action
        """
        self.folded = True

    def reset_bet(self) -> None:
        """
        Reset the player's current bet to zero for the next betting round.

        Side Effects:
            - Resets bet amount to 0
            - Logs the previous bet amount and reset
        """
        previous_bet = self.bet
        self.bet = 0
        PlayerLogger.log_state_reset(self.name, previous_bet, "bet reset")

    def reset_for_new_round(self) -> None:
        """Reset player state for a new round."""
        self.bet = 0
        self.folded = False
        self.is_all_in = False
        self.checked = False
        self.called = False

        PlayerLogger.log_state_reset(self.name, context="new round")

    @property
    def position(self) -> PlayerPosition:
        """Get the player's current position at the table."""
        return self._position

    @position.setter
    def position(self, value: PlayerPosition):
        """Set the player's position at the table."""
        old_position = getattr(self, "_position", None)
        self._position = value
        PlayerLogger.log_position_change(
            self.name, value.name, old_position.name if old_position else None
        )

    def get_state(self) -> PlayerState:
        """Get the current state of this player."""
        return PlayerState.from_player(self)

    def __str__(self) -> str:
        """
        Create a string representation of the player's current state.

        Returns:
            str: A formatted string showing the player's name, chip count,
                 and whether they have folded
        """
        return f"{self.name} (chips: {self.chips}, folded: {self.folded})"

    def __eq__(self, other):
        """Compare two players for equality.

        Players are considered equal if they have the same name.
        """
        if not isinstance(other, Player):
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash function for Player objects.

        This is required when implementing __eq__ to maintain hashability.
        """
        return hash(self.name)

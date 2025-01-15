from dataclasses import dataclass
from typing import Optional


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
    min_bet: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.starting_chips <= 0:
            raise ValueError("Starting chips must be positive")
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.ante < 0:
            raise ValueError("Ante cannot be negative")
        # Set min_bet to big blind if not specified
        if self.min_bet is None:
            self.min_bet = self.big_blind
        elif self.min_bet < self.big_blind:
            raise ValueError("Minimum bet cannot be less than big blind")

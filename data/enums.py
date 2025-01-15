from enum import Enum


class GameType(str, Enum):
    """Valid poker game types."""

    FIVE_CARD_DRAW = "5-card-draw"
    TEXAS_HOLDEM = "texas-holdem"
    TEXAS_HOLDEM_NO_LIMIT = "texas-holdem-no-limit"
    SEVEN_CARD_STUD = "7-card-stud"


class ActionType(str, Enum):
    """Valid poker actions."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all-in"


class PlayerStatus(str, Enum):
    """Player status in a game."""

    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all-in"
    ELIMINATED = "eliminated"


class BettingRound(str, Enum):
    """Betting rounds in poker."""

    PRE_DRAW = "pre-draw"  # For 5-card draw
    POST_DRAW = "post-draw"
    PRE_FLOP = "pre-flop"  # For Hold'em
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class MessageInterpretation(str, Enum):
    """Valid message interpretations."""

    TRUST = "trust"
    IGNORE = "ignore"
    COUNTER_BLUFF = "counter-bluff"


class StrategyStyle(str, Enum):
    """Valid strategy styles."""

    AGGRESSIVE = "Aggressive Bluffer"
    CAUTIOUS = "Calculated and Cautious"
    CHAOTIC = "Chaotic and Unpredictable"


class BettingLimit(str, Enum):
    """Valid betting limits."""

    NO_LIMIT = "no-limit"

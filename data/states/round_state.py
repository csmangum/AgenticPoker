from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator


class RoundPhase(str, Enum):
    """Represents the different phases of a poker round."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    PRE_DRAW = "pre_draw"
    POST_DRAW = "post_draw"


class RoundState(BaseModel):
    """Represents the state of a betting round."""

    phase: RoundPhase
    current_bet: int
    round_number: int
    raise_count: int = 0
    dealer_position: Optional[int] = None
    small_blind_position: Optional[int] = None
    big_blind_position: Optional[int] = None
    first_bettor_index: Optional[int] = None
    main_pot: int = 0
    side_pots: List[Dict[str, Any]] = []

    # Add fields for tracking betting actions
    last_raiser: Optional[str] = None
    last_aggressor: Optional[str] = None
    needs_to_act: List[str] = []
    acted_this_phase: List[str] = []
    is_complete: bool = False
    winner: Optional[str] = None

    @validator("current_bet", "round_number", "raise_count", "main_pot")
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    @classmethod
    def new_round(cls, round_number: int) -> "RoundState":
        """Create a new round state for the start of a hand."""
        return cls(
            phase=RoundPhase.PREFLOP,
            current_bet=0,
            round_number=round_number,
            raise_count=0,
        )

    def update_phase(self, new_phase: RoundPhase) -> None:
        """Update the phase of the round."""
        self.phase = new_phase

    def track_action(self, player_name: str, action: str) -> None:
        """Track player actions during the round."""
        if action in ["raise", "bet"]:
            self.last_raiser = player_name
            self.raise_count += 1
        if action in ["raise", "bet", "call"]:
            self.last_aggressor = player_name
        self.acted_this_phase.append(player_name)
        if player_name in self.needs_to_act:
            self.needs_to_act.remove(player_name)

    def reset_for_new_phase(self) -> None:
        """Reset tracking fields for a new phase of the round."""
        self.needs_to_act = []
        self.acted_this_phase = []
        self.last_raiser = None
        self.last_aggressor = None
        self.is_complete = False
        self.winner = None

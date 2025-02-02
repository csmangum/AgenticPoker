"""
This module provides the Table class which manages a poker table and its players.

The Table class handles:
- Player seating and rotation
- Betting round management
- Player state tracking (active, all-in, folded)
- Turn progression during betting rounds
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

from data.enums import ActionType
from data.types.action_decision import ActionDecision
from game.player import Player
from loggers.table_logger import TableLogger

if TYPE_CHECKING:
    from agents.agent import Agent


class Table:
    """A poker table that manages players and coordinates gameplay.

    The Table represents a poker table where players are seated and betting rounds occur.
    It manages the flow of the game by tracking player states, coordinating betting rounds,
    and ensuring proper turn order.

    Core responsibilities:
    - Maintains player seating and rotation order
    - Tracks player states (active, all-in, folded)
    - Manages betting rounds and player actions
    - Determines when betting rounds are complete
    - Provides player state queries (active, inactive, all-in, folded)

    Attributes:
        players (List[Player]): Ordered list of players at the table, representing seating order
        index (int): Current dealer position in the rotation (0 to len(players)-1)
        needs_to_act (Set[Player]): Set of players who still need to act in current round
        last_raiser (Optional[Player]): The last player who raised in the current round
        current_bet (int): The current bet amount that players need to call
        action_tracking (List[ActionDecision]): List tracking all actions in the current round
    """

    def __init__(self, players: List[Player]):
        """Initialize the table with a list of players.

        Sets up the initial player rotation and tracking sets for betting rounds.
        All players start as needing to act and no players have acted since last raise.

        Args:
            players (List[Player]): Ordered list of Player objects to be included in the table.
                                  The order determines the seating arrangement and turn order.
        """
        self.players = players
        self.initial_players = players.copy()
        self.index = 0
        self.needs_to_act = set(players)  # Track players who still need to act
        self.last_raiser = None
        self.current_bet = 0  # Track the current bet amount
        self.action_tracking = []
        TableLogger.log_table_creation(len(players))

    def update(self, action_decision: ActionDecision, agent: "Agent") -> None:
        """Update the table state based on the agent's action decision."""
        self.mark_player_acted(agent, action_decision)

    def get_next_player(self) -> Optional[Player]:
        """Get the next active player in the rotation who can take an action.

        Cycles through the players list starting from the current index until finding
        an active player (not folded, not all-in). Updates the index for the next call.
        Wraps around to the beginning of the list if reaching the end.

        Returns:
            Optional[Player]: The next active player who can take an action.
                            Returns None if no active players remain.
        """
        if not self.active_players():
            return None

        # Find next active player
        start_index = self.index  # Remember where we started
        while True:
            if self.index >= len(self.players):
                self.index = 0
            player = self.players[self.index]

            # Only increment after we've checked the current player
            if player in self.active_players():
                self.index = (self.index + 1) % len(
                    self.players
                )  # Increment for next time
                TableLogger.log_next_player(
                    player.name, self.index - 1, [p.name for p in self.needs_to_act]
                )
                return player

            self.index = (self.index + 1) % len(self.players)

            # If we've gone full circle without finding an active player
            if self.index == start_index:
                return None

    def is_round_complete(self) -> Tuple[bool, str]:
        """Determine if the current betting round is complete.

        A betting round is complete when either:
        1. All but one player has folded
        2. All remaining active players have:
           - Called the current bet amount, or
           - Gone all-in, or
           - Folded
        3. All players have had at least one chance to act

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if round is complete, False otherwise
                - str: A message explaining why the round is complete or not
        """
        active_players = self.active_players()

        # If only one player remains, round is complete
        if len(active_players) <= 1:
            return True, "only one active player"

        # Check if any player still needs to act
        if self.needs_to_act:
            return False, "players still need to act"

        # Check if all active players have either called or are all-in
        for player in active_players:
            if not player.folded and not player.is_all_in:
                if player.bet < self.current_bet:
                    return False, "not all players have called"

        return True, "betting round complete"

    def get_active_count(self) -> int:
        """Get the number of active players (not folded, not all-in).

        Returns:
            int: Number of active players
        """
        return len(self.active_players())

    def get_all_in_count(self) -> int:
        """Get the number of all-in players.

        Returns:
            int: Number of all-in players
        """
        return len(self.all_in_players())

    def get_folded_count(self) -> int:
        """Get the number of folded players.

        Returns:
            int: Number of folded players
        """
        return len(self.folded_players())

    def remove_player(self, player: Player) -> None:
        """Remove a player from the table."""
        self.players.remove(player)
        TableLogger.log_player_removed(player.name)

    def mark_player_acted(
        self, player: Player, action_decision: ActionDecision
    ) -> None:
        """Mark a player as having acted in the current betting round.

        Updates tracking sets to reflect a player's action. If the action was a raise,
        resets the needs_to_act set and requires all other active players to act again.
        Updates the current bet amount and tracks the action in action_tracking.

        Args:
            player (Player): The player who just completed their action
            action_decision (ActionDecision): The action decision made by the player
        """
        self.needs_to_act.discard(player)

        # Record opponent actions for each agent
        for other_player in self.players:
            if other_player != player and hasattr(
                other_player, "record_opponent_action"
            ):
                bet_amount = None
                if action_decision.action_type == ActionType.RAISE:
                    bet_amount = action_decision.raise_amount
                other_player.record_opponent_action(
                    opponent_name=player.name,
                    action=action_decision.action_type.value,
                    bet_amount=bet_amount,
                )

        if action_decision.action_type == ActionType.RAISE:
            # Update current bet amount
            self.current_bet = action_decision.raise_amount
            self.last_raiser = player
            # Everyone else needs to act again (except folded/all-in players)
            self.needs_to_act = set(p for p in self.active_players() if p != player)

        elif action_decision.action_type == ActionType.CALL:
            player.bet = self.current_bet

        self.action_tracking.append(action_decision)

        TableLogger.log_player_acted(
            player.name,
            action_decision.action_type,
            [p.name for p in self.needs_to_act],
            len(self.action_tracking),
        )

    def reset_action_tracking(self) -> None:
        """Reset the action tracking for a new betting round (street)."""
        self.needs_to_act = set(self.active_players())
        self.current_bet = 0
        self.last_raiser = None

        TableLogger.log_action_tracking_reset([p.name for p in self.active_players()])

    def active_players(self) -> List[Player]:
        """Get the list of active players who can take actions.

        Active players are those who:
        - Have not folded
        - Are not all-in
        - Have chips remaining

        Returns:
            List[Player]: List of players who can still act in the current hand
        """
        active = [p for p in self.players if not p.folded and p.chips > 0]
        TableLogger.log_table_state(
            len(active), len(self.all_in_players()), len(self.folded_players())
        )
        return active

    def inactive_players(self) -> List[Player]:
        """Get the list of players who cannot take actions.

        Inactive players are those who:
        - Have folded
        - Are all-in
        - Have no chips remaining

        Returns:
            List[Player]: List of players who cannot act in the current hand
        """
        return [p for p in self.players if p.folded or p.is_all_in or p.chips == 0]

    def all_in_players(self) -> List[Player]:
        """Get the list of players who are all-in.

        Returns:
            List[Player]: List of players who have committed all their chips
        """
        return [p for p in self.players if p.is_all_in]

    def folded_players(self) -> List[Player]:
        """Get the list of players who have folded.

        Returns:
            List[Player]: List of players who have folded their hands
        """
        return [p for p in self.players if p.folded]

    def __iter__(self):
        """Make Table iterable through all players.

        Yields each player in the table in order, regardless of their state
        (active, all-in, or folded).

        Yields:
            Player: Each player in the table in sequence
        """
        for player in self.players:
            yield player

    def __len__(self) -> int:
        """Get the number of players in the table.

        Returns:
            int: Number of players in the table
        """
        return len(self.players)

    def __getitem__(self, index: int) -> Player:
        """Get a player from the table by index.

        Args:
            index (int): The index of the player to retrieve

        Returns:
            Player: The player at the specified index
        """
        return self.players[index]

    def __contains__(self, player: Player) -> bool:
        """Check if a player is in the table.

        Returns:
            bool: True if the player is in the table, False otherwise
        """
        return player in self.players

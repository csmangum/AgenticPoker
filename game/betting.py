"""Poker betting round management and rules implementation.

This module handles all betting-related logic for a poker game, including:
- Managing complete betting rounds
- Processing individual player actions
- Collecting blinds and antes
- Tracking betting amounts and pot sizes
- Handling all-in situations and side pots

The main components are:
- handle_betting_round: Entry point for managing a complete betting round
- betting_round: Core betting mechanics for a single round
- collect_blinds_and_antes: Handles forced bets at the start of each hand

The module ensures proper poker betting rules are followed:
- Players act in clockwise order
- Betting continues until all active players have:
  * Placed equal bets
  * Folded
  * Gone all-in
- Side pots are created when players go all-in
"""

from typing import TYPE_CHECKING

from loggers.betting_logger import BettingLogger

if TYPE_CHECKING:
    from game.game import Game


def handle_betting_round(game: "Game") -> bool:
    """Manages a complete betting round and updates game state accordingly.

    This function serves as the main entry point for handling a betting round.
    It performs the following:
    1. Input validation (players exist, pot is valid)
    2. Manages the betting process via betting_round()
    3. Updates game state with results
    4. Determines if the game should continue

    Args:
        game: Game object containing all game state and player information

    Returns:
        bool: Whether the game should continue (True if multiple players remain)

    Raises:
        ValueError: If there are no players or if the pot amount is negative

    Side Effects:
        - May modify player chip counts and betting amounts
        - Updates betting state in game.table
    """
    if not game.table:
        raise ValueError("Cannot run betting round with no players")

    # Initialize pot to 0 if None
    if game.pot.pot is None:
        game.pot.pot = 0
    elif game.pot.pot < 0:
        raise ValueError("Pot amount cannot be negative")

    # Run betting round with validated GameState
    betting_round(game)

    # Determine if game should continue
    active_count = sum(1 for p in game.table if not p.folded)
    should_continue = active_count > 1

    return should_continue


def betting_round(game: "Game") -> None:
    """Manages a complete round of betting among active players.

    This function handles the core betting mechanics for a poker round by:
    1. Resetting action tracking for the new round
    2. Processing the betting cycle via _process_betting_cycle()

    The actual processing of player actions and pot management happens in
    _process_betting_cycle().

    Args:
        game: The Game instance containing all game state, including players,
             pot manager, and round state.
    """
    game.table.reset_action_tracking()

    _process_betting_cycle(game)

    # After betting cycle completes, move all bets to pot
    game.pot.end_betting_round(game.table.players)


def _process_betting_cycle(game: "Game") -> None:
    """Process a single cycle of betting for all active players.

    This function manages the core betting loop where each player takes their turn
    to act. It continues until the betting round is complete (all players have acted
    and betting is equalized) or only one player remains.

    The cycle includes:
    1. Getting the next player to act
    2. Logging the current game state
    3. Having the player decide and execute their action
    4. Updating the table state based on the action
    5. Checking if the round is complete

    Args:
        game: The Game instance containing the current game state, including:
             - table: Table object with player and betting state
             - pot: Manages main pot and side pots

    Side Effects:
        - Updates player betting amounts
        - Updates pot size
        - Updates table state (last_raiser, needs_to_act, etc.)
        - Logs betting actions and game state
    """
    complete = False

    while not complete:
        agent = game.table.get_next_player()
        if not agent:
            # No more active players, clear needs_to_act and end round
            game.table.needs_to_act.clear()
            break

        # Skip players who are all-in - they can't act anymore
        if agent.is_all_in:
            continue

        # Add current pot amount to logging
        BettingLogger.log_player_turn(
            player_name=agent.name,
            hand=agent.hand.show() if hasattr(agent, "hand") else "Unknown",
            chips=agent.chips,
            current_bet=agent.bet,
            pot=game.pot.pot,
            active_players=[p.name for p in game.table.players if not p.folded],
            last_raiser=game.table.last_raiser.name if game.table.last_raiser else None,
        )

        action_decision = agent.decide_action(game)
        agent.execute(action_decision, game)

        # Update table state based on action
        game.table.update(action_decision, agent)

        complete, reason = game.table.is_round_complete()

        BettingLogger.log_line_break()


def collect_blinds_and_antes(game, dealer_index, small_blind, big_blind, ante):
    """Collects mandatory bets (blinds and antes) from players.

    This function handles the collection of forced bets at the start of a hand:
    - Collects antes from all players (if applicable)
    - Collects small blind from the player after the dealer
    - Collects big blind from the player after the small blind

    Args:
        game: Game instance for updating game state
        dealer_index: Position of the dealer button
        small_blind: Amount of the small blind
        big_blind: Amount of the big blind
        ante: Amount of the ante (0 if no ante)

    Returns:
        int: Total amount collected from blinds and antes
    """
    collected = 0
    num_players = len(game.table)

    # Reset all player bets first
    for player in game.table:
        player.bet = 0

    # Collect antes
    if ante > 0:
        BettingLogger.log_collecting_antes()
        for player in game.table:
            ante_amount = min(ante, player.chips)
            collected += player.place_bet(ante_amount, game)
            BettingLogger.log_blind_or_ante(
                player.name, ante, ante_amount, is_ante=True
            )

    # Small blind
    sb_index = (dealer_index + 1) % num_players
    sb_player = game.table[sb_index]
    sb_amount = min(small_blind, sb_player.chips)
    actual_sb = sb_player.place_bet(sb_amount, game)
    collected += actual_sb

    BettingLogger.log_blind_or_ante(
        sb_player.name, small_blind, actual_sb, is_small_blind=True
    )

    # Big blind
    bb_index = (dealer_index + 2) % num_players
    bb_player = game.table[bb_index]
    bb_amount = min(big_blind, bb_player.chips)
    actual_bb = bb_player.place_bet(bb_amount, game)
    collected += actual_bb

    BettingLogger.log_blind_or_ante(bb_player.name, big_blind, actual_bb)

    BettingLogger.log_line_break()
    return collected

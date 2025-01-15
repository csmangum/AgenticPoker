import logging
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

from config import GameConfig
from data.states.round_state import RoundPhase
from data.types.action_response import ActionResponse, ActionType
from data.types.pot_types import SidePot

from .player import Player

if TYPE_CHECKING:
    from agents.agent import Agent
    from game.game import Game


logger = logging.getLogger(__name__)


def betting_round(
    game: "Game",
) -> Union[int, Tuple[int, List[SidePot]]]:
    """Manages a complete round of betting among active players.

    Args:
        game

    Returns:
        Either:
        - int: The new pot amount (if no side pots were created)
        - Tuple[int, List[SidePot]]: The new pot amount and list of side pots
    """
    active_players: List[Player] = [p for p in game.players if not p.folded]
    all_in_players: List[Player] = [p for p in active_players if p.is_all_in]
    round_complete: bool = False
    last_raiser: Optional[Player] = None

    # Get big blind player based on position
    big_blind_player: Optional[Player] = _get_big_blind_player(game, active_players)

    # Track who needs to act and who has acted since last raise
    needs_to_act: Set[Player] = set(active_players)
    acted_since_last_raise: Set[Player] = set()

    # Get current bet amount
    current_bet: int = _get_current_bet(game)

    # Track the highest bet seen in this round
    highest_bet: int = current_bet

    # Main betting loop
    while not round_complete:

        for agent in active_players:
            logging.info(f"---- {agent.name} is active ----")

            should_skip, reason = _should_skip_player(agent, needs_to_act)
            if should_skip:
                logging.info(f"{agent.name} {reason}, skipping")
                continue

            # Get player action
            action_decision = agent.decide_action(game)

            # Process the action
            pot, new_current_bet, new_last_raiser = _process_player_action(
                agent,
                action_decision,
                current_bet,
                last_raiser,
                active_players,
                all_in_players,
                game,
            )

            # Update highest bet seen
            if new_current_bet > highest_bet:
                highest_bet = new_current_bet

            # Update last raiser
            if new_last_raiser:
                last_raiser = new_last_raiser
                # Reset acted_since_last_raise set
                acted_since_last_raise = {agent}
                # Everyone except the raiser needs to act again
                needs_to_act = set(
                    p
                    for p in active_players
                    if p != agent and not p.folded and p.chips > 0
                )
            else:
                # Add player to acted set and remove from needs_to_act
                acted_since_last_raise.add(agent)
                needs_to_act.discard(agent)

            # Check if betting round should continue
            active_non_folded = set(
                p for p in active_players if not p.folded and p.chips > 0
            )
            all_acted = acted_since_last_raise == active_non_folded

            # If everyone has acted since last raise, give last raiser final chance
            if (
                all_acted
                and last_raiser
                and not last_raiser.folded
                and last_raiser.chips > 0
            ):
                needs_to_act = {last_raiser}
                last_raiser = None  # Reset to prevent infinite loop
            elif all_acted:
                needs_to_act.clear()  # No one else needs to act

        # End round conditions
        if not needs_to_act:
            round_complete = True

        # Or if we only have one or zero players with chips left
        active_with_chips = [p for p in active_players if p.chips > 0]
        if len(active_with_chips) <= 1:
            round_complete = True

    # If any all-ins occurred, compute side pots
    if all_in_players:
        side_pots = calculate_side_pots(active_players, all_in_players)
        return pot, side_pots

    return pot


def handle_betting_round(
    game: "Game",
) -> Tuple[int, Optional[List[SidePot]], bool]:
    """Manages a complete betting round for all players and determines if the
    game should continue.

    Args:
        game: Game object

    Returns:
        Tuple containing:
        - int: New pot amount
        - Optional[List[SidePot]]: Any side pots created
        - bool: Whether the game should continue
    """
    if not game.players:
        raise ValueError("Cannot run betting round with no players")
    if game.pot_manager.pot < 0:
        raise ValueError("Pot amount cannot be negative")
    if any(not isinstance(p, Player) for p in game.players):
        raise TypeError("All elements in players list must be Player instances")

    # Run betting round with validated GameState
    result = betting_round(game)

    # Handle return value based on whether side pots were created
    if isinstance(result, tuple):
        new_pot, side_pots = result
    else:
        new_pot = result
        side_pots = None

    # Update game state
    game.pot_manager.main_pot = new_pot
    if side_pots:
        game.pot_manager.side_pots = [
            {
                "amount": pot.amount,
                "eligible_players": pot.eligible_players,
            }
            for pot in side_pots
        ]

    # Determine if game should continue
    active_count = sum(1 for p in game.players if not p.folded)
    should_continue = active_count > 1

    return new_pot, side_pots, should_continue


def validate_bet_to_call(
    current_bet: int, player_bet: int, is_big_blind: bool = False
) -> int:
    """Calculates the amount a player needs to add to call the current bet.

    Args:
        current_bet: The current bet amount that needs to be matched
        player_bet: The amount the player has already bet in this round
        is_big_blind: Whether the player is in the big blind position

    Returns:
        int: The additional amount the player needs to bet to call.
            Returns 0 for big blind when no raises have occurred.
    """
    bet_to_call = max(
        0, current_bet - player_bet
    )
    return bet_to_call


def _process_player_action(
    player: Player,
    action_decision: ActionResponse,
    current_bet: int,
    last_raiser: Optional[Player],
    active_players: List[Player],
    all_in_players: List[Player],
    game: "Game",
) -> Tuple[int, int, Optional[Player]]:
    """Processes a player's betting action and updates game state accordingly.

    Handles the mechanics of:
    - Processing folds, calls, and raises
    - Managing all-in situations
    - Updating pot and bet amounts
    - Enforcing betting limits
    - Logging actions and state changes

    Args:
        player: Player making the action
        action: The action ('fold', 'call', or 'raise')
        amount: Bet amount (total bet for raises)
        pot: Current pot amount
        current_bet: Current bet to match
        last_raiser: Last player who raised
        active_players: List of players still in hand
        all_in_players: List of players who are all-in
        game: Game object

    Returns:
        Tuple containing:
        - int: Updated pot amount
        - int: New current bet amount
        - Optional[Player]: New last raiser (if any)
    """
    new_last_raiser = None

    # Calculate how much player needs to call, accounting for big blind position
    is_big_blind = player.is_big_blind if hasattr(player, "is_big_blind") else False
    to_call = validate_bet_to_call(current_bet, player.bet, is_big_blind)

    # Log initial state with active player context
    logging.info(
        f"  Active players: {[p.name for p in active_players if not p.folded]}"
    )
    logging.info(f"  Last raiser: {last_raiser.name if last_raiser else 'None'}")
    logging.info(
        f"  Hand: {player.hand.show() if hasattr(player, 'hand') else 'Unknown'}"
    )
    logging.info(f"  Current bet to call: ${to_call}")
    logging.info(f"  Player chips: ${player.chips}")
    logging.info(f"  Player current bet: ${player.bet}")
    logging.info(f"  Current pot: ${game.pot_manager.pot}")

    if action_decision.action_type == ActionType.CHECK:
        logging.info(f"{player.name} checks")
        return game.pot_manager.pot, current_bet, None

    elif action_decision.action_type == ActionType.FOLD:
        player.folded = True
        logging.info(f"{player.name} folds")

    elif action_decision.action_type == ActionType.CALL:
        # Player can only bet what they have
        bet_amount = min(to_call, player.chips)
        actual_bet = player.place_bet(
            bet_amount
        )

        # Add the bet to the pot
        game.pot_manager.pot += actual_bet

        # If they went all-in trying to call, count it as a raise
        if player.chips == 0 and bet_amount < to_call:
            new_last_raiser = player

        status = " (all in)" if player.chips == 0 else ""
        logging.info(f"{player.name} calls ${bet_amount}{status}")
        logging.info(f"  Pot after call: ${game.pot_manager.pot}")

    elif action_decision.action_type == ActionType.RAISE:
        # Calculate minimum raise amount (current bet + minimum raise increment)
        min_raise = current_bet + game.config.min_bet

        # Process valid raise
        if action_decision.raise_amount >= min_raise:
            # Calculate how much more they need to add
            to_add = action_decision.raise_amount - player.bet
            actual_bet = player.place_bet(to_add)
            game.pot_manager.pot += actual_bet

            # Update current bet and raise count
            if action_decision.raise_amount > current_bet:
                current_bet = action_decision.raise_amount
                new_last_raiser = player

            status = " (all in)" if player.chips == 0 else ""
            logging.info(
                f"{player.name} raises to ${action_decision.raise_amount}{status}"
            )
        else:
            # Invalid raise amount, convert to call
            logging.info(
                f"Raise amount ${action_decision.raise_amount} below minimum (${min_raise}), converting to call"
            )
            return _process_call(player, current_bet, game.pot_manager.pot)

    # Update all-in status considering active players
    if player.chips == 0 and not player.folded:
        if player not in all_in_players:
            all_in_players.append(player)
            # Check if this creates a showdown situation
            active_with_chips = [
                p for p in active_players if not p.folded and p.chips > 0
            ]
            if len(active_with_chips) <= 1:
                logging.info("Showdown situation: Only one player with chips remaining")

    # Log updated state
    logging.info(f"  Pot after action: ${game.pot_manager.pot}")
    logging.info(f"  {player.name}'s remaining chips: ${player.chips}")
    logging.info("")

    # Update game state if provided, but DON'T increment raise count here
    if game.round_state:
        game.round_state.current_bet = current_bet
        if new_last_raiser:
            game.round_state.last_raiser = new_last_raiser.name

    return game.pot_manager.pot, current_bet, new_last_raiser


def calculate_side_pots(
    active_players: List[Player], all_in_players: List[Player]
) -> List[SidePot]:
    """Creates side pots when one or more players are all-in.

    Args:
        active_players: List of players still in the hand
        all_in_players: List of players who are all-in

    Returns:
        List[SidePot]: List of side pots, each containing:
            - amount: The amount in this pot
            - eligible_players: Players who can win this pot
    """
    # Sort all-in players by their total bet
    all_in_players.sort(key=lambda p: p.bet)

    # Get all active players who haven't folded
    active_non_folded = [p for p in active_players if not p.folded]

    side_pots = []
    previous_bet = 0

    # Process each bet level (including all-in amounts)
    # Include both all-in players and active players to get all bet levels
    bet_levels = sorted(set(p.bet for p in active_non_folded))

    for bet_level in bet_levels:
        if bet_level > previous_bet:
            # Find eligible players for this level
            # A player is eligible if they bet at least this amount
            eligible_players = [p.name for p in active_non_folded if p.bet >= bet_level]

            # Calculate pot amount for this level
            # Each eligible player contributes the difference between this level and previous level
            level_amount = (bet_level - previous_bet) * len(eligible_players)

            if level_amount > 0:
                side_pots.append(
                    SidePot(amount=level_amount, eligible_players=eligible_players)
                )

            previous_bet = bet_level

    if side_pots:
        logging.debug("Side pots created:")
        for i, pot in enumerate(side_pots, start=1):
            logging.debug(
                f"  Pot {i}: ${pot.amount} - Eligible: {pot.eligible_players}"
            )

    return side_pots


def log_action(
    player: Player, action: str, amount: int = 0, current_bet: int = 0, pot: int = 0
) -> None:
    """
    Log player actions with optional all-in indicators.
    """
    action_str = (
        f"{player.name}'s turn:\n"
        f"  Current bet to call: ${current_bet}\n"
        f"  Player chips: ${player.chips}\n"
        f"  Player current bet: ${player.bet}\n"
        f"  Current pot: ${pot}"
    )

    if action == "fold":
        action_str += f"\n{player.name} folds"
    elif action == "call":
        all_in = amount >= player.chips
        status = " (all in)" if all_in else ""
        action_str += f"\n{player.name} calls ${amount}{status}"
    elif action == "raise":
        all_in = amount >= player.chips
        status = " (all in)" if all_in else ""
        action_str += f"\n{player.name} raises to ${amount}{status}"

    logging.info(action_str)


def collect_blinds_and_antes(players, dealer_index, small_blind, big_blind, ante):
    """Collects mandatory bets (blinds and antes) from players."""
    collected = 0
    num_players = len(players)

    # Collect antes first
    if ante > 0:
        logging.info("\nCollecting antes:")
        for player in players:
            ante_amount = min(ante, player.chips)
            player.chips -= ante_amount
            collected += ante_amount

            status = " (all in)" if player.chips == 0 else ""
            if ante_amount < ante:
                logging.info(
                    f"  {player.name} posts partial ante of ${ante_amount}{status}"
                )
            else:
                logging.info(f" {player.name} posts ante of ${ante_amount}{status}")

        # Add extra line break after antes
        logging.info("")

    # Small blind
    sb_index = (dealer_index + 1) % num_players
    sb_player = players[sb_index]
    sb_amount = min(small_blind, sb_player.chips)
    actual_sb = sb_player.place_bet(sb_amount)
    collected += actual_sb

    status = " (all in)" if sb_player.chips == 0 else ""
    if actual_sb < small_blind:
        logging.info(
            f"{sb_player.name} posts partial small blind of ${actual_sb}{status}"
        )
    else:
        logging.info(f"{sb_player.name} posts small blind of ${actual_sb}{status}")

    # Big blind
    bb_index = (dealer_index + 2) % num_players
    bb_player = players[bb_index]
    bb_amount = min(big_blind, bb_player.chips)
    actual_bb = bb_player.place_bet(bb_amount)
    collected += actual_bb

    status = " (all in)" if bb_player.chips == 0 else ""
    if actual_bb < big_blind:
        logging.info(
            f"{bb_player.name} posts partial big blind of ${actual_bb}{status}"
        )
    else:
        logging.info(f"{bb_player.name} posts big blind of ${actual_bb}{status}")

    # Add extra line break after blinds
    logging.info("")

    return collected


def _process_call(
    player: Player, current_bet: int, pot: int
) -> Tuple[int, int, Optional[Player]]:
    """Process a call action from a player.

    Args:
        player: Player making the call
        current_bet: Current bet amount to match
        pot: Current pot amount

    Returns:
        Tuple containing:
        - int: Updated pot amount
        - int: Current bet (unchanged)
        - Optional[Player]: Last raiser (None for calls)
    """
    # Calculate how much more they need to add to call
    to_call = current_bet - player.bet
    bet_amount = min(to_call, player.chips)

    # Place the bet
    actual_bet = player.place_bet(bet_amount)
    pot += actual_bet

    # Log the action
    status = " (all in)" if player.chips == 0 else ""
    logger.info(f"{player.name} calls ${bet_amount}{status}")
    logger.info(f"  Pot after call: ${pot}")

    return pot, current_bet, None

    # Reset flags after round
    if big_blind_player:
        big_blind_player.is_big_blind = False


def _get_big_blind_player(
    game: "Game", active_players: List[Player]
) -> Optional[Player]:
    """Get the big blind player and set their flag.

    Args:
        game: Game instance
        active_players: List of active players

    Returns:
        Optional[Player]: The big blind player if found, None otherwise
    """
    is_preflop = game.round_state and game.round_state.phase == RoundPhase.PREFLOP

    if (
        is_preflop
        and game.round_state
        and game.round_state.big_blind_position is not None
    ):
        bb_index = game.round_state.big_blind_position
        if 0 <= bb_index < len(active_players):
            big_blind_player = active_players[bb_index]
            # Set flag on player object for bet validation
            big_blind_player.is_big_blind = True
            return big_blind_player

    return None


def _get_current_bet(game: "Game") -> int:
    """Get the current bet amount, using game state or default value.

    Args:
        game: Game instance

    Returns:
        int: The current bet amount
    """
    current_bet = game.current_bet
    if current_bet <= 0:
        current_bet = game.min_bet if game else GameConfig.min_bet
    return current_bet


def _should_skip_player(player: Player, needs_to_act: Set[Player]) -> Tuple[bool, str]:
    """Determines if a player should be skipped in the betting round.

    Args:
        player: The player to check
        needs_to_act: Set of players that still need to act

    Returns:
        Tuple[bool, str]: (should_skip, reason)
    """
    if player.folded or player.chips == 0:
        return True, "folded or has no chips"

    if player not in needs_to_act:
        return True, "doesn't need to act"

    return False, ""

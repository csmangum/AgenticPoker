import logging
from unittest.mock import MagicMock, patch

import pytest

from game.card import Card
from game.deck import Deck
from game.draw import handle_draw_phase
from game.hand import Hand
from game.player import Player


@pytest.fixture
def mock_players():
    """Create a list of mock players for testing."""
    players = []
    for i in range(3):
        player = Player(f"Player{i}", 1000)
        player.hand = Hand()
        player.hand.add_cards([Card(rank, "Hearts") for rank in range(2, 7)])
        player.folded = False
        players.append(player)
    return players


@pytest.fixture
def mock_deck():
    """Create a mock deck for testing."""
    deck = Deck()
    # Ensure specific cards for testing
    deck.cards = [Card(rank, "Spades") for rank in range(7, 12)]
    return deck


def test_handle_draw_phase_no_discards(mock_players, mock_deck, caplog):
    """Test draw phase when no players discard cards."""
    caplog.set_level(logging.INFO)

    # None of the players have decide_draw method
    handle_draw_phase(mock_players, mock_deck)

    # Check that all players kept their original hands
    for player in mock_players:
        assert len(player.hand.cards) == 5
        assert all(card.suit == "Hearts" for card in player.hand.cards)
        assert (
            "Non-AI player or player without decision method; keeping current hand"
            in caplog.text
        )


def test_handle_draw_phase_with_discards(mock_players, mock_deck, caplog):
    """Test draw phase when a player discards cards."""
    caplog.set_level(logging.INFO)

    # Set up specific cards in deck for testing
    mock_deck.cards = [Card(suit="Spades", rank="2")]

    # Add decide_draw method to first player
    mock_players[0].decide_draw = MagicMock(return_value=[0])  # Discard first card

    handle_draw_phase(mock_players, mock_deck)

    # Check first player's hand was modified
    assert len(mock_players[0].hand.cards) == 5
    assert mock_players[0].hand.cards[-1].suit == "Spades"
    assert all(card.suit == "Hearts" for card in mock_players[0].hand.cards[:-1])

    # Check other players' hands remained unchanged
    for player in mock_players[1:]:
        assert len(player.hand.cards) == 5
        assert all(card.suit == "Hearts" for card in player.hand.cards)


def test_handle_draw_phase_reshuffle(mock_players, mock_deck, caplog):
    """Test draw phase when deck needs reshuffling."""
    caplog.set_level(logging.INFO)

    # Empty the deck except for one card
    mock_deck.cards = [Card(suit="Spades", rank="2")]

    # Add decide_draw method to first player to discard 3 cards
    mock_players[0].decide_draw = MagicMock(return_value=[0, 1, 2])
    original_cards = mock_players[0].hand.cards.copy()

    handle_draw_phase(mock_players, mock_deck)

    # Verify reshuffle behavior with new message format
    assert (
        "Deck low on cards (1 remaining). Need 3 cards. Reshuffling..." in caplog.text
    )
    assert len(mock_players[0].hand.cards) == 5

    # After reshuffling, we can only verify:
    # 1. The number of cards is correct
    # 2. Some of the original cards are still in the hand
    # 3. At least one card from the deck was used

    # Check that at least one Spades card was drawn
    spades_count = sum(
        1 for card in mock_players[0].hand.cards if card.suit == "Spades"
    )
    assert spades_count > 0, "Should have drawn at least one card from the deck"

    # Check that some original Hearts cards remain
    hearts_count = sum(
        1 for card in mock_players[0].hand.cards if card.suit == "Hearts"
    )
    assert hearts_count > 0, "Should have kept some original Hearts cards"

    # Verify total cards is still correct
    assert hearts_count + spades_count == 5, "Total cards should be 5"


def test_handle_draw_phase_folded_players(mock_players, mock_deck):
    """Test draw phase skips folded players."""
    # Make second player folded
    mock_players[1].folded = True
    mock_players[1].decide_draw = MagicMock()  # Should never be called

    handle_draw_phase(mock_players, mock_deck)

    mock_players[1].decide_draw.assert_not_called()


def test_handle_draw_phase_no_discard_decision(mock_players, mock_deck, caplog):
    """Test draw phase when player decides not to discard."""
    caplog.set_level(logging.INFO)

    # Add decide_draw method that returns empty list
    mock_players[0].decide_draw = MagicMock(return_value=[])
    original_hand = mock_players[0].hand.cards.copy()

    handle_draw_phase(mock_players, mock_deck)

    assert mock_players[0].hand.cards == original_hand
    # Update to match actual log message
    assert "No cards discarded; keeping current hand" in caplog.text


def test_handle_draw_phase_multiple_discards(mock_players, mock_deck, caplog):
    """Test draw phase with multiple players discarding."""
    caplog.set_level(logging.DEBUG)

    # Create enough cards to avoid reshuffle
    new_cards = [Card(suit="Spades", rank=str(r)) for r in range(2, 12)]  # 10 cards
    mock_deck.cards = new_cards.copy()

    logging.debug(f"Deck cards before draw: {[str(c) for c in mock_deck.cards]}")

    # Set up discards for two players
    mock_players[0].decide_draw = MagicMock(return_value=[0])
    mock_players[1].decide_draw = MagicMock(return_value=[1, 2])

    # Store original cards that won't be discarded
    player0_kept = mock_players[0].hand.cards[1:].copy()
    player1_kept = [mock_players[1].hand.cards[0]] + mock_players[1].hand.cards[
        3:
    ].copy()

    logging.debug(
        f"Player 0 original hand: {[str(c) for c in mock_players[0].hand.cards]}"
    )
    logging.debug(
        f"Player 1 original hand: {[str(c) for c in mock_players[1].hand.cards]}"
    )

    handle_draw_phase(mock_players, mock_deck)

    logging.debug(
        f"Player 0 final hand: {[str(c) for c in mock_players[0].hand.cards]}"
    )
    logging.debug(
        f"Player 1 final hand: {[str(c) for c in mock_players[1].hand.cards]}"
    )
    logging.debug(f"Deck cards after draw: {[str(c) for c in mock_deck.cards]}")

    # Check first player's hand
    assert len(mock_players[0].hand.cards) == 5
    assert mock_players[0].hand.cards[:-1] == player0_kept  # First 4 cards unchanged
    assert mock_players[0].hand.cards[-1].suit == "Spades"  # New card should be a Spade

    # Check second player's hand
    assert len(mock_players[1].hand.cards) == 5
    assert mock_players[1].hand.cards[0] == player1_kept[0]  # First card unchanged
    assert mock_players[1].hand.cards[1:3] == player1_kept[1:]  # Middle cards unchanged
    assert all(
        card.suit == "Spades" for card in mock_players[1].hand.cards[-2:]
    )  # Last two should be Spades


def test_handle_draw_phase_negative_index(mock_players, mock_deck, caplog):
    """Test draw phase when player provides negative discard indexes."""
    caplog.set_level(logging.INFO)

    # Add decide_draw method that returns negative index
    mock_players[0].decide_draw = MagicMock(return_value=[-1])
    original_hand = mock_players[0].hand.cards.copy()

    handle_draw_phase(mock_players, mock_deck)

    # Check that hand remained unchanged
    assert mock_players[0].hand.cards == original_hand
    assert "invalid discard indexes" in caplog.text
    assert f"{mock_players[0].name} keeping current hand" in caplog.text


def test_draw_phase_logging_not_duplicated(mock_players, mock_deck, caplog):
    """Test that draw phase logging isn't duplicated."""
    caplog.set_level(logging.INFO)

    # Set up a player with discards
    mock_players[0].decide_draw = MagicMock(return_value=[0, 1])

    handle_draw_phase(mock_players, mock_deck)

    # Count occurrences of discard logging for this player
    discard_logs = sum(
        1
        for record in caplog.records
        if f"Draw phase: {mock_players[0].name} discarding" in record.message
    )

    assert discard_logs == 1, "Discard action should only be logged once"

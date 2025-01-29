import pytest

from game.card import Card
from game.evaluator import evaluate_hand


@pytest.fixture
def royal_flush():
    """Royal flush in spades"""
    return [
        Card("A", "♠"),
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("10", "♠"),
    ]


@pytest.fixture
def straight_flush():
    """9-high straight flush in hearts"""
    return [
        Card("9", "♥"),
        Card("8", "♥"),
        Card("7", "♥"),
        Card("6", "♥"),
        Card("5", "♥"),
    ]


@pytest.fixture
def four_of_a_kind():
    """Four aces"""
    return [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("A", "♣"),
        Card("K", "♠"),
    ]


@pytest.fixture
def full_house():
    """Aces full of kings"""
    return [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("K", "♠"),
        Card("K", "♥"),
    ]


@pytest.fixture
def flush():
    """Ace-high flush in diamonds"""
    return [
        Card("A", "♦"),
        Card("J", "♦"),
        Card("8", "♦"),
        Card("6", "♦"),
        Card("2", "♦"),
    ]


@pytest.fixture
def straight():
    """Ace-high straight"""
    return [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
        Card("10", "♠"),
    ]


@pytest.fixture
def three_of_a_kind():
    """Three aces"""
    return [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("K", "♠"),
        Card("Q", "♥"),
    ]


@pytest.fixture
def two_pair():
    """Aces and kings"""
    return [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("K", "♠"),
        Card("Q", "♥"),
    ]


@pytest.fixture
def one_pair():
    """Pair of aces"""
    return [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("Q", "♠"),
        Card("J", "♥"),
    ]


@pytest.fixture
def high_card():
    """Ace high"""
    return [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♠"),
        Card("9", "♥"),
    ]


def test_royal_flush(royal_flush):
    """Test royal flush evaluation"""
    rank, tiebreakers, description = evaluate_hand(royal_flush)
    assert rank == 1
    assert tiebreakers == [14, 13, 12, 11, 10]
    assert description == "Royal Flush"


def test_straight_flush(straight_flush):
    """Test straight flush evaluation"""
    rank, tiebreakers, description = evaluate_hand(straight_flush)
    assert rank == 2
    assert tiebreakers == [9, 8, 7, 6, 5]
    assert description == "Straight Flush, 9 high"


def test_four_of_a_kind(four_of_a_kind):
    """Test four of a kind evaluation"""
    rank, tiebreakers, description = evaluate_hand(four_of_a_kind)
    assert rank == 3
    assert tiebreakers == [14, 13]  # Four aces with king kicker
    assert description == "Four of a Kind, 14s"


def test_full_house(full_house):
    """Test full house evaluation"""
    rank, tiebreakers, description = evaluate_hand(full_house)
    assert rank == 4
    assert tiebreakers == [14, 13]  # Aces full of kings
    assert description == "Full House, 14s over 13s"


def test_flush(flush):
    """Test flush evaluation"""
    rank, tiebreakers, description = evaluate_hand(flush)
    assert rank == 5
    assert tiebreakers == [14, 11, 8, 6, 2]  # Ace-high flush
    assert description == "Flush, 14 high"


def test_straight(straight):
    """Test straight evaluation"""
    rank, tiebreakers, description = evaluate_hand(straight)
    assert rank == 6
    assert tiebreakers == [14, 13, 12, 11, 10]  # Ace-high straight
    assert description == "Straight, 14 high"


def test_three_of_a_kind(three_of_a_kind):
    """Test three of a kind evaluation"""
    rank, tiebreakers, description = evaluate_hand(three_of_a_kind)
    assert rank == 7
    assert tiebreakers == [14, 13, 12]  # Three aces with K,Q kickers
    assert description == "Three of a Kind, 14s"


def test_two_pair(two_pair):
    """Test two pair evaluation"""
    rank, tiebreakers, description = evaluate_hand(two_pair)
    assert rank == 8
    assert tiebreakers == [14, 13, 12]  # Aces and kings with queen kicker
    assert description == "Two Pair, 14s and 13s"


def test_one_pair(one_pair):
    """Test one pair evaluation"""
    rank, tiebreakers, description = evaluate_hand(one_pair)
    assert rank == 9
    assert tiebreakers == [14, 13, 12, 11]  # Pair of aces with K,Q,J kickers
    assert description == "One Pair, 14s"


def test_high_card(high_card):
    """Test high card evaluation"""
    rank, tiebreakers, description = evaluate_hand(high_card)
    assert rank == 10
    assert tiebreakers == [14, 13, 12, 11, 9]  # A,K,Q,J,9
    assert description == "High Card, 14"


def test_ace_low_straight():
    """Test ace-low straight (A,2,3,4,5)"""
    hand = [
        Card("A", "♠"),
        Card("2", "♥"),
        Card("3", "♦"),
        Card("4", "♣"),
        Card("5", "♠"),
    ]
    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 6
    assert tiebreakers == [5, 4, 3, 2, 1]  # 5-high straight
    assert description == "Straight, 5 high"


def test_ace_low_straight_flush():
    """Test ace-low straight flush"""
    hand = [
        Card("A", "♠"),
        Card("2", "♠"),
        Card("3", "♠"),
        Card("4", "♠"),
        Card("5", "♠"),
    ]
    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 2
    assert tiebreakers == [5, 4, 3, 2, 1]
    assert description == "Straight Flush, 5 high"


def test_invalid_hand_size():
    """Test error handling for invalid hand size"""
    hand = [Card("A", "♠"), Card("K", "♠")]  # Only 2 cards
    with pytest.raises(ValueError, match="Hand must contain exactly 5 cards"):
        evaluate_hand(hand)


def test_hand_comparison_tiebreakers():
    """Test that tiebreakers work correctly for same-rank hands"""
    # Two different two-pairs
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),  # Pair of aces
        Card("K", "♦"),
        Card("K", "♠"),  # Pair of kings
        Card("Q", "♥"),  # Queen kicker
    ]

    hand2 = [
        Card("A", "♦"),
        Card("A", "♣"),  # Pair of aces
        Card("Q", "♣"),
        Card("Q", "♦"),  # Pair of queens
        Card("K", "♣"),  # King kicker
    ]

    rank1, tiebreakers1, _ = evaluate_hand(hand1)
    rank2, tiebreakers2, _ = evaluate_hand(hand2)

    assert rank1 == rank2  # Both are two pair
    assert tiebreakers1 > tiebreakers2  # First hand should win (higher second pair)


def test_flush_tiebreaker():
    """Test that flush tiebreakers are ordered correctly"""
    flush1 = [
        Card("A", "♠"),
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("9", "♠"),
    ]

    flush2 = [
        Card("A", "♥"),
        Card("K", "♥"),
        Card("Q", "♥"),
        Card("J", "♥"),
        Card("8", "♥"),
    ]

    _, tiebreakers1, _ = evaluate_hand(flush1)
    _, tiebreakers2, _ = evaluate_hand(flush2)

    assert tiebreakers1 > tiebreakers2  # First flush should win (higher 5th card)


def test_full_house_tiebreaker():
    """Test full house tiebreaker comparison"""
    # Aces full of kings vs kings full of aces
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("K", "♠"),
        Card("K", "♥"),
    ]

    hand2 = [
        Card("K", "♦"),
        Card("K", "♣"),
        Card("K", "♥"),
        Card("A", "♣"),
        Card("A", "♦"),
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Aces full should beat kings full


def test_four_of_a_kind_tiebreaker():
    """Test four of a kind tiebreaker comparison"""
    # Four aces vs four kings
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("A", "♣"),
        Card("2", "♠"),
    ]

    hand2 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("K", "♣"),
        Card("A", "♠"),
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Four aces should beat four kings


def test_straight_tiebreaker():
    """Test straight tiebreaker comparison"""
    # Ace-high vs king-high straight
    hand1 = [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
        Card("10", "♠"),
    ]

    hand2 = [
        Card("K", "♦"),
        Card("Q", "♣"),
        Card("J", "♥"),
        Card("10", "♦"),
        Card("9", "♠"),
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Ace-high straight should win


def test_three_of_kind_tiebreaker():
    """Test three of a kind tiebreaker comparison"""
    # Three aces vs three kings
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("2", "♣"),
        Card("3", "♠"),
    ]

    hand2 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("A", "♣"),
        Card("Q", "♠"),
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Three aces should beat three kings


def test_one_pair_kicker_tiebreaker():
    """Test one pair with kicker tiebreaker"""
    # Pair of aces with K,Q,J vs pair of aces with K,Q,10
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("Q", "♣"),
        Card("J", "♠"),
    ]

    hand2 = [
        Card("A", "♦"),
        Card("A", "♣"),
        Card("K", "♣"),
        Card("Q", "♦"),
        Card("10", "♠"),
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Higher kicker should win


def test_mixed_suits_straight():
    """Test straight with mixed suits is not a straight flush"""
    hand = [
        Card("10", "♠"),
        Card("9", "♥"),
        Card("8", "♦"),
        Card("7", "♣"),
        Card("6", "♠"),
    ]
    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 6  # Should be straight, not straight flush
    assert tiebreakers == [10, 9, 8, 7, 6]
    assert description == "Straight, 10 high"


def test_same_rank_different_suits():
    """Test that suits don't affect non-flush hand rankings"""
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("K", "♠"),
        Card("Q", "♥"),
    ]

    hand2 = [
        Card("A", "♦"),
        Card("A", "♣"),
        Card("K", "♣"),
        Card("K", "♥"),
        Card("Q", "♦"),
    ]

    rank1, tiebreakers1, _ = evaluate_hand(hand1)
    rank2, tiebreakers2, _ = evaluate_hand(hand2)

    assert rank1 == rank2
    assert tiebreakers1 == tiebreakers2  # Same ranks should have same tiebreakers


def test_four_of_a_kind_kicker_comparison():
    """Test four of a kind with different kickers"""
    hand1 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("K", "♣"),
        Card("A", "♠"),  # Ace kicker
    ]

    hand2 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("K", "♣"),
        Card("Q", "♠"),  # Queen kicker
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same quads, higher kicker should win


def test_full_house_same_three_different_pair():
    """Test full houses with same three of a kind but different pairs"""
    hand1 = [
        Card("Q", "♠"),
        Card("Q", "♥"),
        Card("Q", "♦"),
        Card("K", "♠"),
        Card("K", "♥"),  # Queens full of Kings
    ]

    hand2 = [
        Card("Q", "♣"),
        Card("Q", "♠"),
        Card("Q", "♥"),
        Card("J", "♦"),
        Card("J", "♠"),  # Queens full of Jacks
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same trips, higher pair should win


def test_two_pair_same_high_pair():
    """Test two pairs with same high pair but different second pairs"""
    hand1 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("K", "♠"),
        Card("2", "♥"),  # Aces and Kings
    ]

    hand2 = [
        Card("A", "♦"),
        Card("A", "♣"),
        Card("Q", "♣"),
        Card("Q", "♦"),
        Card("K", "♣"),  # Aces and Queens
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same high pair, higher second pair wins


def test_three_of_kind_kicker_order():
    """Test three of a kind with different kicker combinations"""
    hand1 = [
        Card("Q", "♠"),
        Card("Q", "♥"),
        Card("Q", "♦"),
        Card("A", "♠"),
        Card("K", "♥"),  # Queens with A,K kickers
    ]

    hand2 = [
        Card("Q", "♣"),
        Card("Q", "♠"),
        Card("Q", "♥"),
        Card("K", "♦"),
        Card("J", "♠"),  # Queens with K,J kickers
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same trips, higher kicker should win


def test_high_card_all_different():
    """Test high card hands with all different cards"""
    hand1 = [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
        Card("9", "♠"),  # A,K,Q,J,9
    ]

    hand2 = [
        Card("A", "♦"),
        Card("K", "♣"),
        Card("Q", "♥"),
        Card("J", "♠"),
        Card("8", "♦"),  # A,K,Q,J,8
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same high cards until last card


def test_straight_flush_vs_regular_flush():
    """Test straight flush beats regular flush with higher cards"""
    straight_flush = [
        Card("8", "♥"),
        Card("7", "♥"),
        Card("6", "♥"),
        Card("5", "♥"),
        Card("4", "♥"),
    ]

    higher_flush = [
        Card("A", "♠"),
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("9", "♠"),
    ]

    rank1, _, _ = evaluate_hand(straight_flush)
    rank2, _, _ = evaluate_hand(higher_flush)

    assert rank1 < rank2  # Straight flush (rank 2) beats flush (rank 5)


def test_duplicate_cards():
    """Test that duplicate cards are handled appropriately"""
    hand = [
        Card("A", "♠"),
        Card("A", "♠"),  # Duplicate card
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
    ]

    # This should probably raise an error or handle duplicates in some way
    # Depending on how you want to handle this case
    with pytest.raises(ValueError):
        evaluate_hand(hand)


def test_duplicate_cards_different_suits():
    """Test that cards with same rank but different suits are allowed"""
    hand = [
        Card("A", "♠"),
        Card("A", "♥"),  # Same rank, different suit is valid
        Card("K", "♦"),
        Card("Q", "♣"),
        Card("J", "♠"),
    ]

    # Should not raise an error
    rank, _, _ = evaluate_hand(hand)
    assert rank == 9  # One pair


def test_duplicate_cards_multiple():
    """Test handling of multiple duplicate cards"""
    hand = [
        Card("A", "♠"),
        Card("A", "♠"),  # First duplicate
        Card("K", "♥"),
        Card("K", "♥"),  # Second duplicate
        Card("Q", "♦"),
    ]

    with pytest.raises(ValueError) as exc_info:
        evaluate_hand(hand)
    assert "Duplicate card found" in str(exc_info.value)


def test_duplicate_cards_all_same():
    """Test hand with all same card"""
    hand = [Card("A", "♠")] * 5  # Five ace of spades

    with pytest.raises(ValueError) as exc_info:
        evaluate_hand(hand)
    assert "Duplicate card found" in str(exc_info.value)


def test_wheel_straight_vs_six_high():
    """Test that A-5 straight (wheel) loses to 6-high straight"""
    wheel = [
        Card("A", "♠"),
        Card("2", "♥"),
        Card("3", "♦"),
        Card("4", "♣"),
        Card("5", "♠"),
    ]

    six_high = [
        Card("6", "♦"),
        Card("5", "♣"),
        Card("4", "♥"),
        Card("3", "♠"),
        Card("2", "♦"),
    ]

    _, tiebreakers1, _ = evaluate_hand(wheel)
    _, tiebreakers2, _ = evaluate_hand(six_high)

    assert tiebreakers2 > tiebreakers1  # 6-high should beat 5-high (wheel)


def test_two_pair_same_pairs_different_kicker():
    """Test two pair hands with identical pairs but different kickers"""
    hand1 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("Q", "♣"),
        Card("A", "♠"),  # Ace kicker
    ]

    hand2 = [
        Card("K", "♦"),
        Card("K", "♣"),
        Card("Q", "♠"),
        Card("Q", "♥"),
        Card("2", "♦"),  # Deuce kicker
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Same two pair, higher kicker wins


def test_full_house_same_pair_different_trips():
    """Test full houses with same pair but different three of a kind"""
    hand1 = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("Q", "♣"),
        Card("Q", "♠"),  # Kings full of Queens
    ]

    hand2 = [
        Card("J", "♦"),
        Card("J", "♣"),
        Card("J", "♥"),
        Card("Q", "♦"),
        Card("Q", "♥"),  # Jacks full of Queens
    ]

    _, tiebreakers1, _ = evaluate_hand(hand1)
    _, tiebreakers2, _ = evaluate_hand(hand2)

    assert tiebreakers1 > tiebreakers2  # Higher trips wins regardless of pair


def test_straight_with_face_cards():
    """Test straight with all face cards"""
    hand = [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
        Card("10", "♠"),
    ]

    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 6  # Straight
    assert tiebreakers == [14, 13, 12, 11, 10]
    assert description == "Straight, 14 high"


def test_flush_all_same_suit_sequential():
    """Test flush that looks like straight flush but isn't sequential"""
    hand = [
        Card("A", "♠"),
        Card("Q", "♠"),
        Card("10", "♠"),
        Card("8", "♠"),
        Card("6", "♠"),
    ]

    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 5  # Should be flush, not straight flush
    assert tiebreakers == [14, 12, 10, 8, 6]
    assert description == "Flush, 14 high"


def test_three_pair_scenario():
    """Test hand that could form three pairs (impossible in real play)"""
    hand = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("K", "♦"),
        Card("K", "♣"),
        Card("Q", "♠"),
    ]

    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 8  # Should be treated as two pair
    assert tiebreakers == [14, 13, 12]  # Aces and Kings with Queen kicker
    assert description == "Two Pair, 14s and 13s"


def test_high_card_sequential_not_straight():
    """Test high card hand that's almost a straight"""
    hand = [
        Card("A", "♠"),
        Card("K", "♥"),
        Card("Q", "♦"),
        Card("J", "♣"),
        Card("9", "♠"),  # 9 breaks the straight
    ]

    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 10  # Should be high card
    assert tiebreakers == [14, 13, 12, 11, 9]
    assert description == "High Card, 14"


def test_almost_royal_flush():
    """Test hand that's one card off from royal flush"""
    hand = [
        Card("A", "♠"),
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("9", "♠"),  # 9 instead of 10
    ]

    rank, tiebreakers, description = evaluate_hand(hand)
    assert rank == 5  # Should be flush
    assert tiebreakers == [14, 13, 12, 11, 9]
    assert description == "Flush, 14 high"


def test_four_of_a_kind_same_rank_kicker():
    """Test four of a kind where kicker matches four of a kind rank"""
    # This is impossible in real play but tests the logic
    hand = [
        Card("K", "♠"),
        Card("K", "♥"),
        Card("K", "♦"),
        Card("K", "♣"),
        Card("K", "♠"),  # Duplicate card
    ]

    with pytest.raises(ValueError) as exc_info:
        evaluate_hand(hand)
    assert "Duplicate card found" in str(exc_info.value)


def test_wheel_straight_flush_vs_regular_straight_flush():
    """Test that wheel straight flush loses to higher straight flush"""
    wheel_sf = [
        Card("5", "♠"),
        Card("4", "♠"),
        Card("3", "♠"),
        Card("2", "♠"),
        Card("A", "♠"),
    ]

    higher_sf = [
        Card("6", "♥"),
        Card("5", "♥"),
        Card("4", "♥"),
        Card("3", "♥"),
        Card("2", "♥"),
    ]

    _, tiebreakers1, _ = evaluate_hand(wheel_sf)
    _, tiebreakers2, _ = evaluate_hand(higher_sf)

    assert tiebreakers2 > tiebreakers1  # 6-high should beat 5-high wheel


def test_invalid_card_rank():
    """Test that invalid card ranks are caught"""
    hand = [
        Card("Z", "♠"),  # Invalid rank
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("10", "♠"),
    ]

    with pytest.raises(KeyError):
        evaluate_hand(hand)

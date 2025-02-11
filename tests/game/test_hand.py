import copy

import pytest

from data.types.hand_rank import HandRank
from game.card import Card
from game.hand import Hand


@pytest.fixture
def sample_cards():
    """Fixture providing a list of sample cards."""
    return [
        Card("A", "♠"),
        Card("K", "♠"),
        Card("Q", "♠"),
        Card("J", "♠"),
        Card("10", "♠"),
    ]


@pytest.fixture
def royal_flush(sample_cards):
    """Fixture providing a royal flush hand."""
    return Hand(sample_cards)


@pytest.fixture
def straight_flush():
    """Fixture providing a 9-high straight flush."""
    cards = [
        Card("9", "♥"),
        Card("8", "♥"),
        Card("7", "♥"),
        Card("6", "♥"),
        Card("5", "♥"),
    ]
    return Hand(cards)


class TestHand:
    def test_initialization(self):
        """Test hand initialization with and without cards."""
        # Test empty initialization
        empty_hand = Hand()
        assert len(empty_hand.cards) == 0
        assert empty_hand._rank is None

        # Test initialization with cards
        cards = [Card("A", "♠"), Card("K", "♠")]
        hand = Hand(cards)
        assert len(hand.cards) == 2
        assert hand.cards == cards
        assert hand._rank is None

    def test_comparison_operations(self, royal_flush, straight_flush):
        """Test all comparison operations between hands."""
        # Royal flush should beat straight flush
        assert royal_flush > straight_flush
        assert straight_flush < royal_flush
        assert not (royal_flush < straight_flush)
        assert not (straight_flush > royal_flush)

        # Test equality with identical hands
        same_royal = Hand(royal_flush.cards[:])
        assert royal_flush == same_royal
        assert not (royal_flush != same_royal)

        # Test comparison with empty hand
        empty_hand = Hand()
        assert royal_flush > empty_hand
        assert empty_hand < royal_flush

    def test_add_cards(self, sample_cards):
        """Test adding cards to a hand."""
        hand = Hand()

        # Test adding non-list
        with pytest.raises(TypeError, match="Cards must be provided as a list"):
            hand.add_cards(sample_cards[0])

        # Add single card
        hand.add_cards([sample_cards[0]])
        assert len(hand.cards) == 1
        assert hand._rank is None

        # Add multiple cards
        hand.add_cards(sample_cards[1:])
        assert len(hand.cards) == 5
        assert hand._rank is None

    def test_get_rank_caching(self, royal_flush):
        """Test that hand ranking is cached properly."""
        # First evaluation should cache the result
        initial_rank = royal_flush._get_rank()
        cached_rank = royal_flush._rank

        assert cached_rank is not None
        assert initial_rank == cached_rank

        # Subsequent calls should use cached value
        second_rank = royal_flush._get_rank()
        assert second_rank == initial_rank

        # Adding cards should invalidate cache
        royal_flush.add_cards([Card("2", "♣")])
        assert royal_flush._rank is None

    def test_show_method(self, royal_flush):
        """Test the show method's output format."""
        display = royal_flush.show()

        # Should be multi-line with cards and evaluation
        assert isinstance(display, str)
        assert "A of ♠" in display  # Updated format
        assert "K of ♠" in display  # Updated format
        assert "Royal Flush" in display
        assert "[Rank:" in display
        assert "Tiebreakers:" in display

    def test_empty_hand_display(self):
        """Test that empty hands are displayed correctly."""
        empty_hand = Hand()
        rank, tiebreakers, description = empty_hand._get_rank()
        assert rank == float("inf")
        assert tiebreakers == []
        assert description == "Empty hand"  # Updated expectation
        assert empty_hand.show() == "Empty hand"  # Updated expectation

    def test_evaluate_method(self, royal_flush):
        """Test the evaluate method's output format."""
        evaluation = royal_flush.evaluate()

        # Verify the tuple structure
        assert isinstance(evaluation, tuple)
        assert len(evaluation) == 3

        rank, tiebreakers, description = evaluation
        assert isinstance(rank, HandRank)
        assert isinstance(tiebreakers, list)
        assert isinstance(description, str)
        assert "Royal Flush" in description

    def test_rank_invalidation(self, sample_cards):
        """Test that rank is properly invalidated when hand changes."""
        hand = Hand()

        # Empty hand rank
        initial_rank = hand._get_rank()
        assert initial_rank[2] == "Empty hand"  # Updated to match new message

        # Add one card - should be invalid number
        hand.add_cards([sample_cards[0]])
        assert hand._rank is None
        partial_rank = hand._get_rank()
        assert partial_rank[2] == "Invalid number of cards"

        # Add rest of cards - should be valid hand
        hand.add_cards(sample_cards[1:])
        assert hand._rank is None
        full_rank = hand._get_rank()
        assert "Royal Flush" in full_rank[2]

    def test_comparison_with_invalid_types(self, royal_flush):
        """Test comparisons with invalid types."""
        with pytest.raises(AttributeError):
            royal_flush < "not a hand"
        with pytest.raises(AttributeError):
            royal_flush > 42
        with pytest.raises(AttributeError):
            royal_flush == None

    def test_tiebreaker_comparison(self):
        """Test hands with same rank but different tiebreakers."""
        # Create two pair hands with different kickers
        hand1 = Hand(
            [
                Card("A", "♠"),
                Card("A", "♥"),
                Card("K", "♠"),
                Card("K", "♥"),
                Card("Q", "♠"),  # Queen kicker
            ]
        )

        hand2 = Hand(
            [
                Card("A", "♦"),
                Card("A", "♣"),
                Card("K", "♦"),
                Card("K", "♣"),
                Card("J", "♦"),  # Jack kicker
            ]
        )

        assert hand1 > hand2  # Queen kicker beats Jack kicker
        assert hand2 < hand1
        assert hand1 != hand2

    def test_identical_hands(self):
        """Test comparison of identical hands."""
        hand1 = Hand(
            [
                Card("A", "♠"),
                Card("K", "♠"),
                Card("Q", "♠"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )

        hand2 = Hand(
            [
                Card("A", "♠"),
                Card("K", "♠"),
                Card("Q", "♠"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )

        assert not (hand1 < hand2)
        assert not (hand1 > hand2)
        assert hand1 == hand2

    def test_hand_with_no_cards(self):
        """Test behavior of hands with no cards."""
        empty_hand = Hand()
        assert len(empty_hand.cards) == 0
        assert empty_hand.show() == "Empty hand"  # Updated expectation
        rank, tiebreakers, description = empty_hand._get_rank()
        assert description == "Empty hand"  # Updated expectation

    def test_add_duplicate_cards(self):
        """Test adding duplicate cards to a hand."""
        hand = Hand()
        duplicate_cards = [Card("A", "♠"), Card("A", "♠")]  # Same card

        # Adding duplicate cards should work at Hand level since validation
        # happens in evaluator
        hand.add_cards(duplicate_cards)
        assert len(hand.cards) == 2

        # Rank calculation should handle duplicates appropriately
        rank, _, _ = hand._get_rank()
        assert rank == float("inf")  # Invalid hand due to duplicates

    def test_add_none_or_empty_cards(self):
        """Test adding None or empty list of cards."""
        hand = Hand()

        # Test adding None
        with pytest.raises(TypeError):
            hand.add_cards(None)

        # Test adding empty list
        hand.add_cards([])
        assert len(hand.cards) == 0

        # Test adding list with None
        with pytest.raises(TypeError):
            hand.add_cards([None])

    def test_comparison_with_partial_hands(self):
        """Test comparing hands with fewer than 5 cards."""
        hand1 = Hand([Card("A", "♠"), Card("K", "♠")])  # 2 cards
        hand2 = Hand([Card("A", "♥"), Card("K", "♥"), Card("Q", "♥")])  # 3 cards

        # Both should be ranked as invalid hands
        rank1, _, _ = hand1._get_rank()
        rank2, _, _ = hand2._get_rank()
        assert rank1 == float("inf")
        assert rank2 == float("inf")

        # Comparison should still work
        assert not (hand1 > hand2)
        assert not (hand2 > hand1)
        assert hand1 == hand2  # Both are equally invalid

    def test_oversized_hand(self):
        """Test hand with more than 5 cards."""
        cards = [
            Card("A", "♠"),
            Card("K", "♠"),
            Card("Q", "♠"),
            Card("J", "♠"),
            Card("10", "♠"),
            Card("9", "♠"),
        ]
        hand = Hand(cards)

        # Should still allow creation but rank as invalid
        assert len(hand.cards) == 6
        rank, _, _ = hand._get_rank()
        assert rank == float("inf")

    def test_invalid_card_objects(self):
        """Test handling of invalid card objects."""
        hand = Hand()

        # Test adding non-Card objects
        with pytest.raises(TypeError, match="All elements must be Card objects"):
            hand.add_cards(["not a card"])

        with pytest.raises(TypeError, match="All elements must be Card objects"):
            hand.add_cards([123])

    def test_mixed_case_ranks(self):
        """Test hands with mixed case ranks."""
        hand = Hand(
            [
                Card("a", "♠"),  # Lowercase
                Card("K", "♠"),  # Uppercase
                Card("q", "♠"),  # Lowercase
                Card("J", "♠"),  # Uppercase
                Card("10", "♠"),
            ]
        )

        # Should still evaluate as royal flush
        rank, _, description = hand._get_rank()
        assert "Royal Flush" in description

    def test_unicode_suit_handling(self):
        """Test handling of different Unicode suit representations."""
        # Test with HTML entities
        hand1 = Hand(
            [
                Card("A", "&spades;"),
                Card("K", "&spades;"),
                Card("Q", "&spades;"),
                Card("J", "&spades;"),
                Card("10", "&spades;"),
            ]
        )

        # Test with Unicode symbols
        hand2 = Hand(
            [
                Card("A", "♠"),
                Card("K", "♠"),
                Card("Q", "♠"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )

        # Both should evaluate as royal flush
        _, _, desc1 = hand1._get_rank()
        _, _, desc2 = hand2._get_rank()
        assert "Royal Flush" in desc1
        assert "Royal Flush" in desc2

    def test_comparison_chain(self):
        """Test transitive property of hand comparisons."""
        hand1 = Hand(
            [
                Card("A", "♠"),
                Card("A", "♥"),
                Card("A", "♦"),
                Card("K", "♠"),
                Card("Q", "♠"),
            ]
        )  # Three aces
        hand2 = Hand(
            [
                Card("K", "♠"),
                Card("K", "♥"),
                Card("K", "♦"),
                Card("Q", "♠"),
                Card("J", "♠"),
            ]
        )  # Three kings
        hand3 = Hand(
            [
                Card("Q", "♠"),
                Card("Q", "♥"),
                Card("Q", "♦"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )  # Three queens

        # Verify transitive property: if a > b and b > c, then a > c
        assert hand1 > hand2
        assert hand2 > hand3
        assert hand1 > hand3

    def test_deep_copy_behavior(self):
        """Test that hands can be properly deep copied."""
        original = Hand(
            [
                Card("A", "♠"),
                Card("K", "♠"),
                Card("Q", "♠"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )

        # Deep copy should create new hand with new card objects
        copied = copy.deepcopy(original)

        assert original == copied  # Should be equal in value
        assert original is not copied  # Should be different objects
        assert original.cards is not copied.cards  # Should have different card lists
        assert (
            original.cards[0] is not copied.cards[0]
        )  # Should have different card objects

    def test_memory_management(self):
        """Test memory management with large number of hands."""
        import sys

        # Create a reference hand
        ref_hand = Hand(
            [
                Card("A", "♠"),
                Card("K", "♠"),
                Card("Q", "♠"),
                Card("J", "♠"),
                Card("10", "♠"),
            ]
        )

        # Get initial memory size
        initial_size = sys.getsizeof(ref_hand)

        # Create many hands
        hands = []
        for _ in range(1000):
            hands.append(copy.deepcopy(ref_hand))

        # Verify reasonable memory usage
        total_size = sum(sys.getsizeof(hand) for hand in hands)
        assert total_size < initial_size * 1100  # Allow some overhead

    def test_remove_cards(self):
        """Test removing cards from hand."""
        cards = [
            Card("A", "♠"),
            Card("K", "♠"),
            Card("Q", "♠"),
            Card("J", "♠"),
            Card("10", "♠"),
        ]
        hand = Hand(cards)

        # Remove cards from positions 1 and 3
        hand.remove_cards([1, 3])
        assert len(hand.cards) == 3
        assert hand._rank is None

        # Compare card values instead of objects directly
        assert hand.cards[0].rank == "A" and hand.cards[0].suit == "♠"
        assert hand.cards[1].rank == "Q" and hand.cards[1].suit == "♠"
        assert hand.cards[2].rank == "10" and hand.cards[2].suit == "♠"

    def test_hand_rank_updates_after_draw(self):
        """Test that hand ranks are properly updated when cards change."""

        # Test Case 1: High Card becomes Pair
        initial_hand = Hand(
            [
                Card("10", "♠"),
                Card("8", "♦"),
                Card("6", "♠"),
                Card("4", "♥"),
                Card("K", "♥"),
            ]
        )

        # Verify initial hand rank
        rank, tiebreakers, description = initial_hand.evaluate()
        assert rank == HandRank.HIGH_CARD
        assert "High Card" in description
        assert tiebreakers[0] == 13  # King high

        # Simulate draw by removing cards and adding new ones
        initial_hand.cards = [
            Card("10", "♠"),  # Kept
            Card("Q", "♣"),  # New
            Card("K", "♣"),  # New
            Card("9", "♦"),  # New
            Card("K", "♥"),  # Kept
        ]

        # Force rank recalculation by clearing cache
        initial_hand._rank = None

        # Verify updated hand rank
        rank, tiebreakers, description = initial_hand.evaluate()
        assert rank == HandRank.ONE_PAIR
        assert "Pair" in description
        assert tiebreakers[0] == 13  # Pair of Kings

        # Test Case 2: Pair becomes High Card
        pair_hand = Hand(
            [
                Card("9", "♣"),
                Card("5", "♣"),
                Card("9", "♠"),
                Card("8", "♥"),
                Card("J", "♦"),
            ]
        )

        # Verify initial pair
        rank, tiebreakers, description = pair_hand.evaluate()
        assert rank == HandRank.ONE_PAIR
        assert "Pair" in description
        assert tiebreakers[0] == 9  # Pair of 9s

        # Simulate draw by replacing cards
        pair_hand.cards = [
            Card("6", "♥"),  # New
            Card("K", "♦"),  # New
            Card("9", "♠"),  # Kept
            Card("8", "♥"),  # Kept
            Card("J", "♦"),  # Kept
        ]

        # Force rank recalculation
        pair_hand._rank = None

        # Verify updated hand rank
        rank, tiebreakers, description = pair_hand.evaluate()
        assert rank == HandRank.HIGH_CARD
        assert "High Card" in description
        assert tiebreakers[0] == 13  # King high

        # Test Case 3: Pair of 3s becomes High Card
        three_pair_hand = Hand(
            [
                Card("3", "♠"),
                Card("4", "♠"),
                Card("3", "♣"),
                Card("2", "♠"),
                Card("K", "♠"),
            ]
        )

        # Verify initial pair
        rank, tiebreakers, description = three_pair_hand.evaluate()
        assert rank == HandRank.ONE_PAIR
        assert "Pair" in description
        assert tiebreakers[0] == 3  # Pair of 3s

        # Simulate draw
        three_pair_hand.cards = [
            Card("7", "♦"),  # New
            Card("4", "♦"),  # New
            Card("3", "♣"),  # Kept
            Card("2", "♠"),  # Kept
            Card("K", "♠"),  # Kept
        ]

        # Force rank recalculation
        three_pair_hand._rank = None

        # Verify updated hand rank
        rank, tiebreakers, description = three_pair_hand.evaluate()
        assert rank == HandRank.HIGH_CARD
        assert "High Card" in description
        assert tiebreakers[0] == 13  # King high

    def test_one_pair_kicker_comparison(self):
        """Test that one pair hands are compared correctly with kickers."""
        # Charlie's hand: A♥, A♠, K♦, 9♦, 8♦
        # Pair of Aces with K,9,8 kickers
        charlie_hand = Hand(
            [
                Card("A", "♥"),
                Card("A", "♠"),
                Card("K", "♦"),
                Card("9", "♦"),
                Card("8", "♦"),
            ]
        )

        # Alice's hand: Q♠, A♦, 7♠, A♣, 10♠
        # Pair of Aces with Q,10,7 kickers
        alice_hand = Hand(
            [
                Card("Q", "♠"),
                Card("A", "♦"),
                Card("7", "♠"),
                Card("A", "♣"),
                Card("10", "♠"),
            ]
        )

        # Direct comparison
        assert (
            charlie_hand > alice_hand
        ), "Charlie's K kicker should beat Alice's Q kicker"
        assert (
            alice_hand < charlie_hand
        ), "Alice's Q kicker should lose to Charlie's K kicker"
        assert not (alice_hand > charlie_hand), "Alice's hand should not beat Charlie's"

        # Compare using compare_to method
        assert (
            charlie_hand.compare_to(alice_hand) > 0
        ), "compare_to should show Charlie's hand is better"
        assert (
            alice_hand.compare_to(charlie_hand) < 0
        ), "compare_to should show Alice's hand is worse"

        # Verify the actual rankings
        charlie_rank, charlie_tiebreakers, _ = charlie_hand.evaluate()
        alice_rank, alice_tiebreakers, _ = alice_hand.evaluate()

        assert (
            charlie_rank == alice_rank == HandRank.ONE_PAIR
        ), "Both should be one pair"
        assert (
            charlie_tiebreakers > alice_tiebreakers
        ), "Charlie's kickers should be higher"

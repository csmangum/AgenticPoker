import pytest
from game.player import Player
from game.hand import Hand


class TestPlayer:
    @pytest.fixture
    def player(self):
        return Player("TestPlayer", 1000)

    def test_player_initialization(self, player):
        """Test that a player is initialized with correct default values"""
        assert player.name == "TestPlayer"
        assert player.chips == 1000
        assert player.bet == 0
        assert player.folded is False
        assert isinstance(player.hand, Hand)

    def test_place_bet_normal(self, player):
        """Test placing a valid bet"""
        bet_amount = 500
        actual_bet = player.place_bet(bet_amount)

        assert actual_bet == bet_amount
        assert player.chips == 500  # Started with 1000
        assert player.bet == 500

    def test_place_bet_more_than_chips(self, player):
        """Test placing a bet larger than available chips"""
        bet_amount = 1500  # More than initial 1000
        actual_bet = player.place_bet(bet_amount)

        assert actual_bet == 1000  # Should only bet what's available
        assert player.chips == 0
        assert player.bet == 1000

    def test_place_negative_bet(self, player):
        """Test that placing a negative bet raises ValueError"""
        with pytest.raises(ValueError):
            player.place_bet(-100)

    def test_fold(self, player):
        """Test folding functionality"""
        assert player.folded is False
        player.fold()
        assert player.folded is True

    def test_reset_bet(self, player):
        """Test resetting bet amount"""
        player.place_bet(500)
        assert player.bet == 500

        player.reset_bet()
        assert player.bet == 0

    def test_reset_for_new_round(self, player):
        """Test resetting player state for new round"""
        # Setup: place bet and fold
        player.place_bet(500)
        player.fold()
        assert player.bet == 500
        assert player.folded is True

        # Reset for new round
        player.reset_for_new_round()
        assert player.bet == 0
        assert player.folded is False
        # Chips should remain unchanged
        assert player.chips == 500

    def test_str_representation(self, player):
        """Test string representation of player"""
        expected = "TestPlayer (chips: 1000, folded: False)"
        assert str(player) == expected

    def test_multiple_bets_same_round(self, player):
        """Test placing multiple bets in the same round"""
        player.place_bet(300)
        player.place_bet(200)

        assert player.chips == 500
        assert player.bet == 500

    def test_bet_all_chips(self, player):
        """Test betting all available chips"""
        actual_bet = player.place_bet(1000)

        assert actual_bet == 1000
        assert player.chips == 0
        assert player.bet == 1000

    def test_zero_bet(self, player):
        """Test placing a zero bet"""
        actual_bet = player.place_bet(0)

        assert actual_bet == 0
        assert player.chips == 1000
        assert player.bet == 0

    def test_initialize_with_zero_chips(self):
        """Test creating a player with zero initial chips"""
        player = Player("BrokePlayer", 0)

        assert player.chips == 0
        # Verify can't bet anything
        actual_bet = player.place_bet(100)
        assert actual_bet == 0
        assert player.bet == 0

    def test_initialize_with_negative_chips(self):
        """Test that initializing with negative chips raises ValueError"""
        with pytest.raises(ValueError):
            Player("NegativePlayer", -100)

    def test_multiple_folds(self, player):
        """Test that folding multiple times keeps player folded"""
        player.fold()
        assert player.folded is True

        player.fold()  # Second fold
        assert player.folded is True

    def test_bet_after_fold(self, player):
        """Test that a player can still bet after folding (shouldn't be prevented)"""
        player.fold()
        actual_bet = player.place_bet(500)

        assert actual_bet == 500
        assert player.chips == 500
        assert player.bet == 500
        assert player.folded is True

    def test_decimal_chip_amount(self):
        """Test that initializing with decimal chip amount raises ValueError"""
        with pytest.raises(ValueError):
            Player("DecimalPlayer", 100.5)

    def test_reset_bet_multiple_times(self, player):
        """Test resetting bet multiple times"""
        player.place_bet(500)
        player.reset_bet()
        player.reset_bet()  # Second reset

        assert player.bet == 0
        assert player.chips == 500

    def test_name_empty_string(self):
        """Test creating a player with an empty name"""
        with pytest.raises(ValueError):
            Player("", 1000)

    def test_name_whitespace(self):
        """Test creating a player with whitespace name"""
        with pytest.raises(ValueError):
            Player("   ", 1000)

    def test_very_large_chip_amount(self):
        """Test initialization and betting with very large chip amounts"""
        large_amount = 10**9  # 1 billion
        player = Player("RichPlayer", large_amount)

        bet_amount = 10**8  # 100 million
        actual_bet = player.place_bet(bet_amount)

        assert actual_bet == bet_amount
        assert player.chips == large_amount - bet_amount
        assert player.bet == bet_amount
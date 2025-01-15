import unittest
from game.game import AgenticPoker
from game.player import Player
from game.evaluator import evaluate_texas_holdem_hand
from game.betting import handle_betting_round
from game.config import GameConfig
from data.enums import GameType, BettingLimit


class TestTexasHoldemNoLimit(unittest.TestCase):
    def setUp(self):
        self.players = ["Alice", "Bob", "Charlie"]
        self.config = GameConfig(
            small_blind=10,
            big_blind=20,
            ante=0,
            min_bet=20,
            max_rounds=10,
        )
        self.game = AgenticPoker(self.players, config=self.config)

    def test_game_flow(self):
        self.game.start_game()
        self.assertEqual(len(self.game.players), 3)
        self.assertEqual(self.game.round_number, 1)

    def test_no_limit_betting_round(self):
        self.game.start_round()
        new_pot, side_pots, should_continue = handle_betting_round(self.game)
        self.assertTrue(should_continue)
        self.assertEqual(new_pot, 60)  # Small blind + Big blind
        self.assertIsNone(side_pots)

    def test_evaluate_texas_holdem_hand(self):
        from game.card import Card

        hole_cards = [Card("A", "♠"), Card("K", "♠")]
        community_cards = [
            Card("Q", "♠"),
            Card("J", "♠"),
            Card("10", "♠"),
            Card("2", "♦"),
            Card("3", "♣"),
        ]
        evaluation = evaluate_texas_holdem_hand(hole_cards, community_cards)
        self.assertEqual(evaluation.rank, 1)
        self.assertEqual(evaluation.description, "Royal Flush")


if __name__ == "__main__":
    unittest.main()

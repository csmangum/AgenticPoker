from game import PokerGame
from game.ai_player import AIPlayer
from util import setup_logging

setup_logging()

# Create AI players with different strategies
players = [
    AIPlayer("Alice", chips=1000, strategy_style="Aggressive Bluffer"),
    AIPlayer("Bob", chips=1000, strategy_style="Calculated and Cautious"),
    AIPlayer("Charlie", chips=1000, strategy_style="Chaotic and Unpredictable"),
    AIPlayer("Dana", chips=1000, strategy_style="Aggressive Bluffer"),
]

# Add ante to make the game more dynamic and give sitting out players a chance to recover
game = PokerGame(players, starting_chips=1000, small_blind=50, big_blind=100, ante=10)
game.start_game()

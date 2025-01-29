from unittest.mock import Mock, patch

import pytest

from agents.agent import Agent
from data.types.action_decision import ActionDecision, ActionType
from data.types.discard_decision import DiscardDecision
from game.card import Card
from game.evaluator import HandEvaluation


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.query.return_value = "DECISION: call"
    return client


@pytest.fixture
def basic_agent(mock_llm_client):
    """Create a basic agent for testing."""
    with patch("agents.agent.LLMClient", return_value=mock_llm_client), patch(
        "agents.agent.ChromaMemoryStore"
    ) as mock_memory, patch("game.player.Player.perceive") as mock_perceive:
        # Setup mock memory store
        mock_store = mock_memory.return_value
        mock_store.get_relevant_memories.return_value = []
        mock_store.add_memory.return_value = None
        
        # Setup mock perceive
        mock_perceive.return_value = {"timestamp": 123456789}

        agent = Agent(
            name="TestAgent",
            chips=1000,
            strategy_style="Aggressive Bluffer",
            use_reasoning=True,
            use_reflection=True,
            use_planning=True,
            session_id="test_session",
        )
        # Mock memory-related methods
        agent.get_relevant_memories = mock_store.get_relevant_memories
        agent._create_memory_query = Mock(return_value="test query")
        return agent


@pytest.fixture
def mock_hand():
    """Create a mock hand with cards."""
    return [Card("A", "♠"), Card("K", "♠"), Card("Q", "♠")]


class TestAgent:
    def test_initialization(self, basic_agent):
        """Test agent initialization.

        Assumptions:
        - Agent is initialized with default communication_style="Intimidating"
        - Agent is initialized with default emotional_state="confident"
        - Personality traits are initialized as a dictionary with default values
        """
        assert basic_agent.name == "TestAgent"
        assert basic_agent.chips == 1000
        assert basic_agent.strategy_style == "Aggressive Bluffer"
        assert basic_agent.use_reasoning is True
        assert basic_agent.use_reflection is True
        assert basic_agent.use_planning is True
        assert basic_agent.communication_style == "Intimidating"
        assert basic_agent.emotional_state == "confident"
        assert isinstance(basic_agent.personality_traits, dict)

    def test_basic_decision(self, basic_agent, mock_llm_client):
        """Test basic decision making.

        Assumptions:
        - LLMResponseGenerator.generate_action returns valid ActionDecision objects
        - Minimum bet validation always succeeds
        - Game state can be mocked with simple string
        - Hand evaluation has valid rank, description and tiebreakers
        """
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"
        mock_hand_eval = Mock(spec=HandEvaluation)
        mock_hand_eval.rank = 5
        mock_hand_eval.description = "Flush"
        mock_hand_eval.tiebreakers = [14, 13, 12, 11, 10]

        # Mock the bet validation functions
        with patch("agents.agent.get_min_bet") as mock_min_bet, patch(
            "agents.agent.validate_bet_amount"
        ) as mock_validate_bet:

            mock_min_bet.return_value = 20
            mock_validate_bet.side_effect = (
                lambda x, y: x
            )  # Return the input amount unchanged

            # Test call decision
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                mock_generate.return_value = ActionDecision(action_type=ActionType.CALL)
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert isinstance(response, ActionDecision)
                assert response.action_type == ActionType.CALL
                assert response.raise_amount is None

            # Test raise decision
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                mock_generate.return_value = ActionDecision(
                    action_type=ActionType.RAISE, raise_amount=100
                )
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert response.action_type == ActionType.RAISE
                assert response.raise_amount == 100
                mock_validate_bet.assert_called_once_with(100, 20)

            # Test fold decision
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                mock_generate.return_value = ActionDecision(action_type=ActionType.FOLD)
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert response.action_type == ActionType.FOLD
                assert response.raise_amount is None

    def test_decide_discard(self, basic_agent):
        """Test discard decision making.

        Assumptions:
        - Agent has valid cards in hand before discard
        - LLMResponseGenerator.generate_discard returns valid DiscardDecision objects
        - Game state only needs round and pot information for discard decisions
        - Error handling returns empty discard indices list
        """
        mock_game_state = {"round": "draw", "pot": 100}
        basic_agent.hand.cards = [Card("A", "♠"), Card("K", "♠"), Card("Q", "♠")]

        with patch(
            "agents.llm_response_generator.LLMResponseGenerator.generate_discard"
        ) as mock_generate:
            # Test successful discard
            mock_generate.return_value = DiscardDecision(indices=[1, 2])
            decision = basic_agent.decide_discard(mock_game_state)
            assert isinstance(decision, DiscardDecision)
            assert decision.indices == [1, 2]

            # Test error handling
            mock_generate.side_effect = Exception("Discard error")
            decision = basic_agent.decide_discard(mock_game_state)
            assert isinstance(decision, DiscardDecision)
            assert decision.indices == []
            assert "Failed to decide discard" in decision.reasoning

    def test_update_strategy(self, basic_agent, mock_llm_client):
        """Test strategy updates based on game outcomes.

        Assumptions:
        - LLM response of "1" means keep current strategy
        - LLM response of "2" means switch to "Aggressive Bluffer"
        - Game outcome contains winner, chips_won and final_hand information
        - Strategy style can be changed directly
        """
        game_outcome = {
            "winner": "TestAgent",
            "chips_won": 100,
            "final_hand": "Flush",
        }

        # Test strategy change
        mock_llm_client.query.return_value = "2"  # Switch to "Aggressive Bluffer"
        basic_agent.strategy_style = "Calculated and Cautious"
        basic_agent.update_strategy(game_outcome)
        assert basic_agent.strategy_style == "Aggressive Bluffer"

        # Test strategy retention
        mock_llm_client.query.return_value = "1"  # Keep current strategy
        original_strategy = basic_agent.strategy_style
        basic_agent.update_strategy(game_outcome)
        assert basic_agent.strategy_style == original_strategy

    def test_perceive(self, basic_agent):
        """Test perception functionality.

        Assumptions:
        - Parent class's perceive method returns a dict
        - Table history has a maximum length of 10
        - Memory store's add_memory method is called exactly once per perception
        - Opponent messages are stored in table history
        """
        game_state = "Current game state"
        opponent_message = "I raise"

        perception = basic_agent.perceive(game_state, opponent_message)

        assert isinstance(perception, dict)
        assert opponent_message in basic_agent.table_history
        assert len(basic_agent.table_history) <= 10  # Check history limit

        # Test memory store interaction
        basic_agent.memory_store.add_memory.assert_called_once()

    def test_analyze_opponent(self, basic_agent, mock_llm_client):
        """Test opponent analysis functionality.

        Assumptions:
        - Opponent stats are stored in defaultdict format
        - LLM returns valid JSON string for analysis
        - Analysis works with both enabled and disabled opponent modeling
        - Default analysis values are used when opponent modeling is disabled
        """
        opponent_name = "Opponent1"
        game_state = "Current game state"

        # Setup mock opponent stats
        basic_agent.opponent_stats[opponent_name] = {
            "actions": {"raise": 10, "call": 5, "fold": 3},
            "bet_sizes": [10, 20, 30],
            "bluff_attempts": 5,
            "bluff_successes": 2,
            "fold_to_raise_count": 3,
            "raise_faced_count": 10,
            "last_five_actions": ["raise", "call", "fold"],
            "position_stats": {"early": {"raise": 2}},
        }

        # Test successful analysis
        mock_llm_client.query.return_value = """
        {
            "patterns": "aggressive",
            "threat_level": "high",
            "style": "tight-aggressive",
            "weaknesses": ["folds to re-raises"],
            "strengths": ["aggressive betting"],
            "recommended_adjustments": ["trap with strong hands"]
        }
        """

        analysis = basic_agent.analyze_opponent(opponent_name, game_state)
        assert isinstance(analysis, dict)
        assert "patterns" in analysis
        assert analysis["threat_level"] == "high"

        # Test with opponent modeling disabled
        basic_agent.use_opponent_modeling = False
        analysis = basic_agent.analyze_opponent(opponent_name, game_state)
        assert analysis["patterns"] == "unknown"
        assert analysis["threat_level"] == "medium"

    def test_context_manager(self):
        """Test context manager functionality.

        Assumptions:
        - Agent properly implements __enter__ and __exit__ methods
        - Cleanup removes all instance attributes properly
        - ChromaMemoryStore and LLMClient can be mocked
        """
        with patch("agents.agent.LLMClient"), patch("agents.agent.ChromaMemoryStore"):
            with Agent(name="TestAgent", session_id="test") as agent:
                assert isinstance(agent, Agent)
                # Perform some operations
                agent.perception_history = ["test"]
                agent.conversation_history = ["test"]

            # Verify cleanup occurred
            assert not hasattr(agent, "perception_history")
            assert not hasattr(agent, "conversation_history")

    def test_message_generation_error_handling(self, basic_agent, mock_llm_client):
        """Test error handling in message generation.

        Assumptions:
        - Invalid LLM responses return "..." as default message
        - LLM exceptions are caught and return "..." as default message
        - Game state can be mocked with simple string
        """
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"

        # Test missing MESSAGE: prefix
        mock_llm_client.query.return_value = "Invalid response"
        message = basic_agent.get_message(mock_game)
        assert message == "..."

        # Test LLM error
        mock_llm_client.query.side_effect = Exception("LLM Error")
        message = basic_agent.get_message(mock_game)
        assert message == "..."

    def test_get_message(self, basic_agent, mock_llm_client):
        """Test message generation.

        Assumptions:
        - LLM client returns non-empty string response
        - Game state can be mocked with simple string
        - Message generation doesn't require full game context
        """
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"

        mock_llm_client.query.return_value = "I'm feeling lucky!"
        message = basic_agent.get_message(mock_game)

        assert isinstance(message, str)
        assert len(message) > 0

    def test_cleanup(self, basic_agent):
        """Test cleanup functionality.

        Assumptions:
        - All mutable state is stored in instance attributes
        - Cleanup removes all instance attributes properly
        - No external resources need to be cleaned up in this test
        """
        # Add some data to clean up
        basic_agent.perception_history = ["perception1", "perception2"]
        basic_agent.conversation_history = ["conv1", "conv2"]
        basic_agent.opponent_stats = {"player1": {"actions": {}}}

        basic_agent.close()

        assert not hasattr(basic_agent, "perception_history")
        assert not hasattr(basic_agent, "conversation_history")
        assert not hasattr(basic_agent, "opponent_stats")

    def test_error_handling(self, basic_agent, mock_llm_client):
        """Test error handling in decision making."""
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"
        mock_hand_eval = Mock(spec=HandEvaluation)
        mock_hand_eval.rank = 5
        mock_hand_eval.description = "Flush"
        mock_hand_eval.tiebreakers = [14, 13, 12, 11, 10]

        # Mock the bet validation functions
        with patch("agents.agent.get_min_bet") as mock_min_bet, patch(
            "agents.agent.validate_bet_amount"
        ) as mock_validate_bet:

            mock_min_bet.return_value = 20
            mock_validate_bet.side_effect = lambda x, y: x

            # Test LLM response generator error
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                mock_generate.side_effect = Exception("LLM Error")
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert (
                    response.action_type == ActionType.CALL
                )  # Default to CALL on error
                assert response.raise_amount is None
                assert "Failed to decide action" in response.reasoning

            # Test raise validation error
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                # Mock validate_bet_amount to raise an exception
                mock_validate_bet.side_effect = ValueError("Invalid bet amount")
                mock_generate.return_value = ActionDecision(
                    action_type=ActionType.RAISE,
                    raise_amount=100,  # Valid amount that will fail validation
                )
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert (
                    response.action_type == ActionType.CALL
                )  # Default to CALL on invalid raise
                assert response.raise_amount is None

    def test_error_handling_in_decision(self, basic_agent):
        """Test error handling in the decision making process.

        Assumptions:
        - LLM errors result in a CALL action as fallback
        - Invalid raise amounts are handled gracefully
        - Game state validation errors don't crash the agent
        - Basic decision making continues despite strategy planner errors
        """
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"
        mock_hand_eval = Mock(spec=HandEvaluation)

        with patch("agents.agent.get_min_bet") as mock_min_bet, patch(
            "agents.agent.validate_bet_amount"
        ) as mock_validate_bet:
            mock_min_bet.return_value = 20
            mock_validate_bet.side_effect = ValueError("Invalid bet")

            # Test LLM error handling
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action",
                side_effect=Exception("LLM Error"),
            ):
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert response.action_type == ActionType.CALL
                assert "Failed to decide action" in response.reasoning

            # Test invalid raise amount
            with patch(
                "agents.llm_response_generator.LLMResponseGenerator.generate_action"
            ) as mock_generate:
                mock_generate.return_value = ActionDecision(
                    action_type=ActionType.RAISE, raise_amount=100  # Use valid amount
                )
                response = basic_agent._decide_action(mock_game, mock_hand_eval)
                assert response.action_type == ActionType.CALL

    @patch("agents.agent.Agent.update_action_values")
    def test_reward_learning(self, mock_update, basic_agent):
        """Test the reward learning functionality.

        Assumptions:
        - Reward weights are initialized with default values
        - Action values are updated based on rewards
        - Learning rate affects the magnitude of updates
        - Reward learning can be disabled without affecting other functionality
        """
        # Test reward learning when enabled
        basic_agent.use_reward_learning = True

        # Setup initial state
        initial_weights = basic_agent.reward_weights.copy()
        initial_values = basic_agent.action_values.copy()

        # Simulate a successful action with reward
        action_state = {
            "action": "raise",
            "chips_gained": 100,
            "hand_strength": 0.8,
            "position": "late",
        }

        # Add action to history
        basic_agent.action_history.append(
            ("raise", action_state, 50)  # action, state, reward
        )

        # Verify update was called when enabled
        basic_agent.update_action_values()
        mock_update.assert_called_once()

        # Test with reward learning disabled
        basic_agent.use_reward_learning = False
        mock_update.reset_mock()
        basic_agent.update_action_values()
        mock_update.assert_not_called()

    def test_strategy_planner_integration(self, basic_agent):
        """Test integration with strategy planner.

        Assumptions:
        - Strategy planner can be enabled/disabled without errors
        - Planner provides valid strategy recommendations
        - Planning errors don't crash the agent
        - Strategy updates are considered in decision making
        """
        mock_game = Mock()
        mock_game.get_state.return_value = "Current game state"
        
        # Setup valid hand
        basic_agent.hand.cards = [
            Card("A", "♠"), Card("K", "♠"), Card("Q", "♠"), 
            Card("J", "♠"), Card("10", "♠")
        ]

        # Test with planning enabled
        basic_agent.use_planning = True
        with patch(
            "agents.strategy_planner.StrategyPlanner.plan_strategy"
        ) as mock_plan:
            mock_plan.return_value = {"action": "raise", "confidence": 0.8}
            
            decision = basic_agent.decide_action(mock_game)
            assert isinstance(decision, ActionDecision)
            mock_plan.assert_called_once()

        # Test with planning disabled
        basic_agent.use_planning = False
        with patch(
            "agents.strategy_planner.StrategyPlanner.plan_strategy"
        ) as mock_plan:
            decision = basic_agent.decide_action(mock_game)
            assert isinstance(decision, ActionDecision)
            mock_plan.assert_not_called()

    def test_memory_integration(self, basic_agent):
        """Test integration with memory store.

        Assumptions:
        - Memory store properly saves and retrieves memories
        - Memory context influences decision making
        - Memory cleanup happens during agent cleanup
        - Memory operations handle errors gracefully
        """
        # Test memory storage
        game_state = "Current game state"
        basic_agent.memory_store.add_memory.reset_mock()

        basic_agent.perceive(game_state, "Opponent raises")
        basic_agent.memory_store.add_memory.assert_called_once()

        # Test memory retrieval
        basic_agent.memory_store.get_relevant_memories.return_value = [
            {"text": "Opponent bluffed", "metadata": {"type": "perception"}}
        ]

        memories = basic_agent.get_relevant_memories("opponent behavior")
        assert len(memories) > 0
        basic_agent.memory_store.get_relevant_memories.assert_called_once()

        # Test memory cleanup
        basic_agent.close()
        assert not hasattr(basic_agent, "memory_store")

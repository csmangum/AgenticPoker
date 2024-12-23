# LLM Poker Agent Documentation

## Overview
The LLM (Language Learning Model) Agent is an AI poker player that uses natural language processing to make strategic decisions, interpret opponent behavior, and communicate during gameplay. It combines poker domain knowledge with configurable personality traits and cognitive mechanisms.

## Features

### Core Capabilities
- Strategic decision making (fold/call/raise)
- Opponent message interpretation
- Natural language communication
- Adaptive strategy updates
- Hand evaluation and drawing decisions
- Historical perception tracking

### Cognitive Mechanisms
The agent has two optional cognitive mechanisms that can be enabled/disabled:

1. **Chain-of-Thought Reasoning** (`use_reasoning`)
   - Systematic step-by-step analysis of the situation
   - Considers hand strength, strategy, opponent behavior, pot odds, and personality alignment
   - More thorough but computationally intensive

2. **Self-Reflection** (`use_reflection`)
   - Reviews initial decisions for consistency
   - Can revise actions that don't align with strategy/personality
   - Adds an extra layer of strategic coherence

## Configuration

### Initialization Parameters
```python
LLMAgent(
    name: str,                                    # Agent's name
    chips: int = 1000,                           # Starting chips
    strategy_style: Optional[str] = None,        # Playing style
    personality_traits: Optional[Dict] = None,    # Behavioral traits
    max_retries: int = 3,                        # LLM query retries
    retry_delay: float = 1.0,                    # Retry wait time
    use_reasoning: bool = True,                  # Enable reasoning
    use_reflection: bool = True,                 # Enable reflection
)
```

### Strategy Styles
Available poker playing styles:
- `"Aggressive Bluffer"`
- `"Calculated and Cautious"`
- `"Chaotic and Unpredictable"`

### Personality Traits
Configurable traits (0.0 to 1.0):
- `aggression`: Tendency to bet and raise
- `bluff_frequency`: Frequency of bluffing
- `risk_tolerance`: Willingness to take risks

## Usage Examples

### Basic Agent Creation
```python
agent = LLMAgent(
    "Alice",
    chips=1000,
    strategy_style="Aggressive Bluffer"
)
```

### Customized Agent with Specific Traits
```python
agent = LLMAgent(
    "Bob",
    chips=1000,
    strategy_style="Calculated and Cautious",
    personality_traits={
        "aggression": 0.3,
        "bluff_frequency": 0.2,
        "risk_tolerance": 0.4
    },
    use_reasoning=True,
    use_reflection=False
)
```

### Different Cognitive Configurations
```python
# Full cognitive capabilities
strategic_agent = LLMAgent(
    "Alice",
    strategy_style="Calculated and Cautious",
    use_reasoning=True,
    use_reflection=True
)

# Basic agent without advanced cognition
simple_agent = LLMAgent(
    "Bob",
    strategy_style="Aggressive Bluffer",
    use_reasoning=False,
    use_reflection=False
)
```

## Key Methods

### Decision Making
- `get_action(game_state: str, opponent_message: Optional[str]) -> str`
  - Determines next poker action (fold/call/raise)
  - Uses reasoning and reflection if enabled

### Message Handling
- `interpret_message(opponent_message: str) -> str`
  - Analyzes opponent messages
  - Returns interpretation (trust/ignore/counter-bluff)

### Strategy Management
- `update_strategy(game_outcome: Dict[str, Any]) -> None`
  - Adapts strategy based on game results
  - Can switch between different playing styles

### State Management
- `reset_state() -> None`
  - Clears perception and conversation history
  - Maintains strategy and personality

## Performance Considerations

### Computational Impact
- Reasoning mechanism: ~2x base computation time
- Reflection mechanism: ~1.5x base computation time
- Both enabled: ~3x base computation time

### Token Usage
- Basic prompt: ~50 tokens
- With reasoning: ~150 tokens
- With reflection: Additional ~100 tokens

### Response Times
- Basic decision: ~1-2 seconds
- With reasoning: ~2-3 seconds
- With reflection: ~3-4 seconds

## Best Practices

1. **Configuration Selection**
   - Use full cognitive capabilities for strategic depth
   - Disable for faster gameplay or testing
   - Mix configurations to study impact

2. **Strategy Style Selection**
   - Match strategy to personality traits
   - Consider table dynamics
   - Use for creating diverse player pools

3. **Error Handling**
   - Agent falls back to "call" on LLM errors
   - Implements retry mechanism for failed queries
   - Logs all decision-making steps

## Limitations

1. **Response Time**
   - Multiple LLM calls can slow gameplay
   - Network latency affects performance
   - Consider timeout settings

2. **Token Usage**
   - Higher costs with cognitive mechanisms
   - Memory limitations with long histories
   - Balance detail vs. efficiency

3. **Strategy Consistency**
   - May show occasional inconsistent behavior
   - Personality drift over long sessions
   - Requires monitoring and adjustment

## Future Improvements

1. **Planned Features**
   - Improved hand reading capabilities
   - Dynamic personality adaptation
   - Multi-model support

2. **Optimization Opportunities**
   - Batch processing for decisions
   - Caching frequent patterns
   - Reduced token usage

3. **Integration Points**
   - Tournament support
   - Performance analytics
   - Strategy training
"""Prompts used by the LLM agent for various decision-making tasks."""

# Used for basic decision making when planning is disabled
# Variables:
# - strategy_style: Agent's current strategy style (e.g. "Aggressive Bluffer")
# - strategy_prompt: Strategy-specific guidance from StrategyManager
# - game_state: Current game situation including cards, bets, etc.
DECISION_PROMPT = """You are a {strategy_style} poker player. You must respond with exactly one action.

{strategy_prompt}

Current situation:
{game_state}

Respond ONLY in this format:
DECISION: <action> <brief reason>
where <action> must be exactly one of: fold, call, raise

Example responses:
DECISION: fold weak hand against aggressive raise
DECISION: call decent draw with good pot odds
DECISION: raise strong hand in position

What is your decision?
"""

# Used for generating table talk messages
# Variables:
# - strategy_style: Agent's current strategy style
# - game_state: Current game situation
# - communication_style: Agent's current communication style
# - table_history: Recent table history
MESSAGE_PROMPT = """You are a {strategy_style} poker player with a {communication_style} communication style.

Current situation:
{game_state}

Recent table history:
{table_history}

CRITICAL RULES:
1. Start with exactly "MESSAGE: "
2. Maximum 15 words after "MESSAGE: "
3. Stay in character for your communication style
4. No explicit card information
5. Include one of these tones: [confident, nervous, amused, frustrated, thoughtful]

Communication Style Guidelines:
- Intimidating: Use subtle psychological pressure and dominance
- Analytical: Focus on probabilities and logical observations
- Friendly: Keep atmosphere light while masking true intentions

Valid examples:
MESSAGE: [confident] Those pot odds don't look so good for you now
MESSAGE: [thoughtful] Interesting how aggressive the table plays after midnight
MESSAGE: [amused] Math suggests this is profitable, but psychology says otherwise

Invalid examples:
MESSAGE: I have pocket aces  (reveals cards)
MESSAGE: Let's collude against the big stack  (suggests unfair play)
MESSAGE: [angry] You're all terrible players!  (too hostile/negative)

Respond with exactly one message following these rules.
"""

# Used for deciding which cards to discard during draw phase
# Variables:
# - strategy_style: Agent's current strategy style
# - game_state: Current hand state
# - cards: List of 5 card objects representing current hand
DISCARD_PROMPT = """You are a {strategy_style} poker player deciding which cards to discard.

Current situation:
{game_state}

CRITICAL RULES:
1. You MUST include a line starting with "DISCARD:" followed by:
   - [x,y] for multiple positions
   - [x] for single position
   - none for keeping all cards
2. Use ONLY card positions (0-4 from left to right)
3. Maximum 3 cards can be discarded
4. Format must be exactly as shown in examples

Example responses:
ANALYSIS:
Pair of Kings, weak kickers
Should discard both low cards

DISCARD: [0,1]

ANALYSIS:
Strong two pair, keep everything

DISCARD: none

ANALYSIS:
Weak high card only
Discard three cards for new draw

DISCARD: [2,3,4]

Current hand positions:
Card 0: {cards[0]}
Card 1: {cards[1]}
Card 2: {cards[2]}
Card 3: {cards[3]}
Card 4: {cards[4]}

What is your discard decision?
"""

# Used for generating strategic messages with memory context
# Variables:
# - strategy_style: Agent's current strategy style
# - game_state: Current game situation
# - recent_observations: Last 3 game events from perception history
# - memory_context: Relevant memories from long-term storage
# - recent_conversation: Last 5 messages exchanged
STRATEGIC_MESSAGE_PROMPT = """You are a {strategy_style} poker player.
Your response must be a single short message (max 10 words) that fits your style.

Game State: {game_state}
Recent Observations: {recent_observations}
Relevant Memories: {memory_context}
Recent Chat: {recent_conversation}

Example responses:
- "I always bet big with strong hands!"
- "Playing it safe until I see weakness."
- "You can't read my unpredictable style!"

Your table talk message:"""

# Used to analyze and interpret opponent messages
# Variables:
# - strategy_style: Agent's current strategy style
# - opponent_message: Message to interpret
# - recent_history: Last 3 game events for context
INTERPRET_MESSAGE_PROMPT = """You are a {strategy_style} poker player.
Opponent's message: '{opponent_message}'
Recent game history: {recent_history}

Based on your strategy style and the game history:
1. Analyze if they are bluffing, truthful, or misleading
2. Consider their previous behavior patterns
3. Think about how this fits your strategy style

Respond with only: 'trust', 'ignore', or 'counter-bluff'
"""

# Used to generate or update strategic plans
# Variables:
# - strategy_style: Agent's current strategy style
# - game_state: Current game situation
# Returns: JSON formatted plan with approach, reasoning, and thresholds
PLANNING_PROMPT = """You are a {strategy_style} poker player planning your strategy.

Current situation:
{game_state}

Create a strategic plan using this exact format:
{{
    "approach": "<aggressive/balanced/defensive>",
    "reasoning": "<brief explanation>",
    "bet_sizing": "<small/medium/large>",
    "bluff_threshold": <float 0-1>,
    "fold_threshold": <float 0-1>
}}

Example:
{{
    "approach": "aggressive",
    "reasoning": "Strong hand, weak opponents",
    "bet_sizing": "large",
    "bluff_threshold": 0.7,
    "fold_threshold": 0.2
}}
"""

# Used to execute actions based on current plan
# Variables:
# - strategy_style: Agent's current strategy style
# - game_state: Current game situation
# - plan_approach: Current strategic approach (aggressive/balanced/defensive)
# - plan_reasoning: Explanation of current plan
# - bluff_threshold: Current bluffing probability threshold
# - fold_threshold: Current folding probability threshold
EXECUTION_PROMPT = """You are a {strategy_style} poker player following this plan:
Approach: {plan_approach}
Reasoning: {plan_reasoning}

Current situation:
{game_state}

Given your {plan_approach} approach:
1. Evaluate if the situation matches your plan
2. Consider pot odds and immediate action costs
3. Factor in your bluff_threshold ({bluff_threshold}) and fold_threshold ({fold_threshold})

Respond with EXECUTE: <fold/call/raise> and brief reasoning
"""

# Add a new prompt for richer table talk interactions
STRATEGIC_BANTER_PROMPT = """You are a {strategy_style} poker player engaging in table talk.

Current situation:
{game_state}
Your position: {position}
Recent actions: {recent_actions}
Opponent tendencies: {opponent_patterns}

Communication style: {communication_style}
Current tone: {emotional_state}

Generate strategic table talk that:
1. Hints at your strategy without revealing it
2. Responds to recent table dynamics
3. Maintains your personality and style
4. Uses psychological elements appropriately

Format:
MESSAGE: [tone] <message>
INTENT: <hidden strategic purpose>
CONFIDENCE: <level 1-10>

Example:
MESSAGE: [amused] The odds of this working get better every orbit
INTENT: Create doubt about bluffing frequency
CONFIDENCE: 8
"""
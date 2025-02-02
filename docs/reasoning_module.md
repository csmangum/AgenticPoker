# Reasoning Module Documentation

## Overview

The `ReasoningModule` class provides the agent with modular, extensible, and domain-specific reasoning capabilities. It supports symbolic, probabilistic, and hybrid reasoning methods, allowing the agent to analyze, infer, and decide based on available data and its goals.

## Class Definition

### `ReasoningModule`

#### Attributes

- `memory_store`: An instance of `ChromaMemoryStore` used for storing facts, experiences, and contextual data.

#### Methods

- `__init__(self, memory_store: ChromaMemoryStore)`: Initializes the `ReasoningModule` with a memory store.
- `deductive_reasoning(self, facts: List[str], rules: List[str]) -> str`: Performs deductive reasoning based on given facts and rules.
- `inductive_reasoning(self, observations: List[str]) -> str`: Performs inductive reasoning based on given observations.
- `probabilistic_reasoning(self, evidence: Dict[str, float]) -> str`: Performs probabilistic reasoning based on given evidence.
- `store_fact(self, fact: str, metadata: Optional[Dict[str, Any]] = None) -> None`: Stores a fact in the memory store with optional metadata.
- `retrieve_facts(self, query: str, k: int = 3) -> List[Dict[str, Any]]`: Retrieves relevant facts from the memory store based on a query.

## Usage

### Initialization

To use the `ReasoningModule`, you need to initialize it with a `ChromaMemoryStore` instance:

```python
from agents.reasoning_module import ReasoningModule
from data.memory import ChromaMemoryStore

memory_store = ChromaMemoryStore("agent_memory")
reasoning_module = ReasoningModule(memory_store)
```

### Deductive Reasoning

To perform deductive reasoning, provide a list of facts and rules:

```python
facts = ["All humans are mortal.", "Socrates is a human."]
rules = ["If all humans are mortal, and Socrates is a human, then Socrates is mortal."]
result = reasoning_module.deductive_reasoning(facts, rules)
print(result)  # Output: "Deductive reasoning result"
```

### Inductive Reasoning

To perform inductive reasoning, provide a list of observations:

```python
observations = ["The sun has risen in the east every day.", "The sun will rise in the east tomorrow."]
result = reasoning_module.inductive_reasoning(observations)
print(result)  # Output: "Inductive reasoning result"
```

### Probabilistic Reasoning

To perform probabilistic reasoning, provide a dictionary of evidence:

```python
evidence = {"Rain": 0.7, "Sprinkler": 0.3}
result = reasoning_module.probabilistic_reasoning(evidence)
print(result)  # Output: "Probabilistic reasoning result"
```

### Storing Facts

To store a fact in the memory store, use the `store_fact` method:

```python
fact = "Socrates is mortal."
metadata = {"source": "philosophy"}
reasoning_module.store_fact(fact, metadata)
```

### Retrieving Facts

To retrieve relevant facts from the memory store, use the `retrieve_facts` method:

```python
query = "Socrates"
facts = reasoning_module.retrieve_facts(query, k=3)
print(facts)  # Output: List of relevant facts
```

## Conclusion

The `ReasoningModule` class provides a flexible and powerful framework for implementing various reasoning methods in the agent. By leveraging the `ChromaMemoryStore`, it ensures that the agent can store and retrieve contextual data efficiently, enabling more intelligent decision-making.

import pytest
from agents.reasoning_module import ReasoningModule
from data.memory import ChromaMemoryStore


@pytest.fixture
def memory_store():
    """Fixture to create a ChromaMemoryStore instance."""
    return ChromaMemoryStore("test_collection")


@pytest.fixture
def reasoning_module(memory_store):
    """Fixture to create a ReasoningModule instance."""
    return ReasoningModule(memory_store)


def test_deductive_reasoning(reasoning_module):
    """Test deductive reasoning method."""
    facts = ["All humans are mortal", "Socrates is a human"]
    rules = ["If all humans are mortal and Socrates is a human, then Socrates is mortal"]
    result = reasoning_module.deductive_reasoning(facts, rules)
    assert result == "Deductive reasoning result"


def test_inductive_reasoning(reasoning_module):
    """Test inductive reasoning method."""
    observations = ["The sun rises every morning", "The sun rose this morning"]
    result = reasoning_module.inductive_reasoning(observations)
    assert result == "Inductive reasoning result"


def test_probabilistic_reasoning(reasoning_module):
    """Test probabilistic reasoning method."""
    evidence = {"Rain": 0.8, "Cloudy": 0.6}
    result = reasoning_module.probabilistic_reasoning(evidence)
    assert result == "Probabilistic reasoning result"


def test_store_and_retrieve_facts(reasoning_module):
    """Test storing and retrieving facts."""
    fact = "The sky is blue"
    metadata = {"type": "observation"}
    reasoning_module.store_fact(fact, metadata)

    retrieved_facts = reasoning_module.retrieve_facts("sky", k=1)
    assert len(retrieved_facts) == 1
    assert retrieved_facts[0]["text"] == fact
    assert retrieved_facts[0]["metadata"] == metadata

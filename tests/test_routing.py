"""Tests for ticket routing engine."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.routing import (
    ROUTING_MAP, get_department, get_priority,
    validate_routing_map, _INTENT_TO_DEPT,
    HIGH_PRIORITY_INTENTS, MEDIUM_PRIORITY_INTENTS,
)


def test_all_77_intents_mapped():
    """Verify all 77 BANKING77 intents are in the routing map."""
    all_intents = []
    for intents in ROUTING_MAP.values():
        all_intents.extend(intents)
    assert len(set(all_intents)) == 77, f"Expected 77, got {len(set(all_intents))}"


def test_no_duplicate_intents():
    """Verify no intent appears in multiple departments."""
    assert validate_routing_map() is True


def test_get_department_known():
    """Test department lookup for known intents."""
    assert get_department('card_arrival') == 'Card Services'
    assert get_department('lost_or_stolen_card') == 'Card Security'
    assert get_department('exchange_rate') == 'Foreign Exchange'


def test_get_department_unknown():
    """Test fallback for unknown intent."""
    assert get_department('some_unknown_intent') == 'General Support'


def test_priority_high():
    """Test high priority assignment."""
    result = get_priority('lost_or_stolen_card', confidence=0.9)
    assert result in ('URGENT', 'HIGH')


def test_priority_normal():
    """Test normal priority assignment."""
    result = get_priority('card_arrival', confidence=0.9)
    assert result == 'NORMAL'


def test_priority_low_confidence():
    """Test medium priority for low confidence."""
    result = get_priority('card_arrival', confidence=0.1)
    assert result == 'MEDIUM'

"""import pytest
from core.action_parser import ActionParser

def test_parse_json_action():
    parser = ActionParser()
    input_text = '{"action": "search_weather", "parameters": {"location": "Paris"}}'
    result = parser.parse(input_text)
    assert result == {
        "action": "search_weather",
        "parameters": {"location": "Paris"}
    }

def test_parse_text_action():
    parser = ActionParser()
    input_text = "Please search weather in Paris"
    result = parser.parse(input_text)
    assert result["action"] == "search_weather"
    assert "location" in result["parameters"]
    assert result["parameters"]["location"].lower() == "paris"

def test_parse_unknown_text():
    parser = ActionParser()
    input_text = "Do something unknown"
    result = parser.parse(input_text)
    assert result is None"""
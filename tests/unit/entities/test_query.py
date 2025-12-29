import pytest
from langrag.entities.query import Query

def test_query_init():
    q = Query(text="  hello  ")
    assert q.text == "hello"  # str_strip_whitespace=True
    assert q.vector is None

def test_query_frozen():
    q = Query(text="test")
    with pytest.raises(Exception): # ValidationError or standard validation error for immutable
        q.text = "new"

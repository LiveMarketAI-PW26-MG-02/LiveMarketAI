from src.search_engine import search, autocomplete, related

def test_exact_symbol_match():
    results = search("AAPL")
    assert results[0].symbol == "AAPL"

def test_sector_filter():
    results = search("", sector="Financials")
    assert all(r.sector == "Financials" for r in results)

def test_autocomplete():
    results = autocomplete("NV")
    symbols = [r["symbol"] for r in results]
    assert "NVDA" in symbols

def test_related_same_sector():
    results = related("AAPL")
    assert all(r.sector == "Technology" for r in results)
    assert all(r.symbol != "AAPL" for r in results)

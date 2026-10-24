import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, timedelta, timezone
from quant_scanner.news import (NewsItem, source_tier, assess, decide, parse_yf_news, parse_rss, clean)

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
H = lambda h: NOW - timedelta(hours=h)

def test_tiers():
    assert source_tier("", "https://www.sec.gov/Archives/x") == 1
    assert source_tier("Reuters", "") == 2 and source_tier("", "https://www.reuters.com/a") == 2
    assert source_tier("CNBC", "") == 3 and source_tier("blog random", "https://tuitero.xyz/x") == 4

def test_rumor_does_not_veto():
    rumor = [NewsItem("XYZ reportedly facing SEC investigation, sources say", "Foro Bursátil", "https://foro.xyz/a", H(3), ("XYZ",))]
    sc = assess(rumor, NOW)
    assert sc[0].label == "sin confirmar" and sc[0].score < 40
    dec, notes = decide(sc)
    assert dec == "OK" and any("SIN confirmar" in n for n in notes)

def test_sec_8k_bankruptcy_vetoes():
    f = [NewsItem("XYZ Corp 8-K", "sec.gov", "https://www.sec.gov/a", H(5), ("XYZ",), form="8-K", items=("1.03", "9.01"))]
    sc = assess(f, NOW)
    assert sc[0].label == "confirmado" and sc[0].impact == "alto" and sc[0].score >= 90
    dec, notes = decide(sc)
    assert dec == "VETO" and "quiebra" in notes[0]
    old = [NewsItem("XYZ Corp 8-K", "sec.gov", "https://www.sec.gov/a", H(24 * 30), ("XYZ",), form="8-K", items=("1.03",))]
    assert decide(assess(old, NOW))[0] == "OK"                      # un hecho viejo no veta

def test_press_corroboration():
    its = [NewsItem("Acme recalls 2 million vehicles over brake defect", "Reuters", "https://reuters.com/a", H(4), ("ACME",)),
           NewsItem("Acme recalls two million vehicles brake defect", "Bloomberg", "https://bloomberg.com/b", H(3), ("ACME",)),
           NewsItem("Acme recalls 2 million vehicles over brake defect", "Reuters", "https://reuters.com/a2", H(3), ("ACME",))]
    sc = assess(its, NOW)
    assert len(sc) == 1 and sc[0].label == "confirmado (prensa)" and sc[0].n_sources == 2   # Reuters repetido cuenta una vez
    assert decide(sc)[0] == "VETO"
    single = assess(its[:1], NOW)[0]
    assert single.label == "probable" and single.score == 70

def test_earnings_is_watch_not_veto():
    f = [NewsItem("ABC 8-K", "sec.gov", "https://sec.gov/x", H(2), ("ABC",), form="8-K", items=("2.02",))]
    dec, notes = decide(assess(f, NOW))
    assert dec == "AVISO" and "balance" in notes[0]

def test_parsers():
    raw = [{"content": {"title": "<b>Nvidia</b> hits record", "provider": {"displayName": "Reuters"},
                        "canonicalUrl": {"url": "https://reuters.com/z"}, "pubDate": "2026-10-20T12:00:00Z"}},
           {"title": "Old format", "publisher": "CNBC", "link": "https://cnbc.com/q", "providerPublishTime": 1792500000}, {"bad": 1}]
    it = parse_yf_news(raw, "NVDA")
    assert len(it) == 2 and it[0].title == "Nvidia hits record" and it[0].source == "Reuters" and it[1].published.tzinfo
    xml = ("<rss><channel><item><title>Fed announces FOMC statement</title><link>https://federalreserve.gov/a</link>"
           "<pubDate>Wed, 28 Oct 2026 18:00:00 GMT</pubDate></item></channel></rss>")
    r = parse_rss(xml, "federalreserve.gov")
    assert len(r) == 1 and r[0].published.year == 2026 and parse_rss("<<no xml", "x") == []
    assert clean("<script>x</script>  hola &amp; chau") == "x hola & chau"

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("OK ", n)

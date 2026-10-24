import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from quant_scanner.config import EventConfig
from quant_scanner.events import all_events, upcoming, event_risk, parse_ics, refresh_from_ics

ICS = ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nDTSTART;VALUE=DATE:20261110\r\nSUMMARY:Consumer Price Index for Oc\r\n tober 2026\r\nEND:VEVENT\r\n"
       "BEGIN:VEVENT\r\nDTSTART:20261106T133000Z\r\nSUMMARY:Employment Situation for October 2026\r\nEND:VEVENT\r\n"
       "BEGIN:VEVENT\r\nDTSTART;VALUE=DATE:20261201\r\nSUMMARY:Employment Situation of Veterans\r\nEND:VEVENT\r\n"
       "BEGIN:VEVENT\r\nDTSTART;VALUE=DATE:20261202\r\nSUMMARY:State Employment and Unemployment (Monthly) for October 2026\r\nEND:VEVENT\r\nEND:VCALENDAR")

def test_ics():
    df = parse_ics(ICS)
    assert len(df) == 2 and set(df["name"]) == {"Consumer Price Index for October 2026", "Employment Situation for October 2026"}
    assert pd.Timestamp("2026-11-10") in set(df["date"])
    assert parse_ics("basura").empty
    merged = refresh_from_ics(ICS)
    assert (merged["source"] == "BLS").sum() == 0 and (merged["source"] == "Fed").sum() >= 16

def test_calendar_and_risk():
    ev = all_events()
    assert ev["date"].is_monotonic_increasing
    fomc = ev[ev["name"].str.contains("FOMC \\(tasas\\)")]
    assert pd.Timestamp("2026-10-28") in set(fomc["date"]) and pd.Timestamp("2026-12-09") in set(fomc["date"])
    assert all(d.weekday() == 2 for d in fomc["date"])              # las decisiones FOMC caen en miércoles
    up = upcoming(ev, pd.Timestamp("2026-10-20"), 21)
    assert list(up["name"].str[:8])[0].startswith("Decisión")
    cfg = EventConfig()
    mult, flags = event_risk(ev, pd.Timestamp("2026-10-27"), cfg)   # señal el martes -> ejecuta el miércoles de la decisión
    assert mult == 0.5 and flags
    mult, flags = event_risk(ev, pd.Timestamp("2026-10-20"), cfg)
    assert mult == 1.0 and not flags
    assert event_risk(ev, pd.Timestamp("2026-10-27"), EventConfig(enabled=False))[0] == 1.0

if __name__ == "__main__":
    test_ics(); test_calendar_and_risk(); print("OK eventos")

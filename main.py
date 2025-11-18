import os
from datetime import datetime, timedelta
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Utility functions ---------
Color = Literal["purple", "pink", "jp", "mjp"]


def classify_color(odd: float) -> Color:
    if 1.0 <= odd < 10.0:
        return "purple"
    if 10.0 <= odd < 100.0:
        return "pink"
    if 100.0 <= odd < 1000.0:
        return "jp"
    return "mjp"


def parse_time(value: Any) -> datetime:
    """Accepts datetime, ISO string, or HH:MM string and returns datetime.
    If only HH:MM is provided, assumes today's date.
    If an integer is provided, treats it as minutes since epoch (for testing).
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # minutes offset from epoch
        return datetime.utcfromtimestamp(int(value) * 60)
    if isinstance(value, str):
        value = value.strip()
        # Try ISO-like formats
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        for f in fmts:
            try:
                return datetime.strptime(value, f)
            except Exception:
                pass
        # HH:MM
        try:
            today = datetime.utcnow().date()
            hh, mm = value.split(":")
            return datetime(today.year, today.month, today.day, int(hh), int(mm))
        except Exception:
            pass
    raise ValueError(f"Unrecognized time format: {value}")


def minute_only(dt: datetime) -> int:
    return int(dt.timestamp() // 60)


# ---------- Data models ----------
class Event(BaseModel):
    time: Any = Field(..., description="Event time: datetime/ISO/HH:MM/int(minutes)")
    odd: float = Field(..., ge=0)

    @validator("time")
    def _validate_time(cls, v):
        # Allow any type; we'll parse later. Keep as-is here.
        return v


class PredictRequest(BaseModel):
    events: List[Event] = Field(..., description="Historical events in chronological or arbitrary order")


class Signal(BaseModel):
    rule: str
    signal: Literal["pink", "jp/mjp", "50x", "warning"]
    base_time: Optional[datetime] = None
    predicted_times: Optional[List[datetime]] = None
    details: Optional[Dict[str, Any]] = None


# ---------- Core logic ----------
ADDS = [3, 5, 8]  # minutes


def detect_signals(events: List[Event]) -> List[Signal]:
    # Normalize and sort events
    norm = [
        {
            "dt": parse_time(ev.time),
            "minute": minute_only(parse_time(ev.time)),
            "odd": ev.odd,
            "color": classify_color(ev.odd),
        }
        for ev in events
    ]
    norm.sort(key=lambda x: x["dt"])  # chronological

    signals: List[Signal] = []

    # Track pink occurrences (minute and dt)
    pinks = [e for e in norm if e["color"] == "pink"]

    # Rule 7: more than 15 minutes without a pink (between last two events or from last pink to now)
    if len(norm) >= 2:
        # check gaps between consecutive events where there is no pink in that window
        last_pink_minute = None
        for e in norm:
            if e["color"] == "pink":
                last_pink_minute = e["minute"]
            else:
                if last_pink_minute is not None:
                    gap = e["minute"] - last_pink_minute
                    if gap > 15:
                        signals.append(
                            Signal(
                                rule="7",
                                signal="warning",
                                base_time=e["dt"],
                                details={"message": ">15 minutes without a pink → reset"},
                            )
                        )
                        # After warning, reset tracker
                        last_pink_minute = e["minute"]

    # Rule 5 and 6: specific decimal endings on pink odds
    for e in pinks:
        frac = round(e["odd"] - int(e["odd"]), 2)
        # Handle floating precision by formatting to 2 decimals
        frac_str = f"{frac:.2f}"
        if frac_str in {"0.87", "0.17", "0.38"}:
            signals.append(
                Signal(
                    rule="5",
                    signal="jp/mjp",
                    base_time=e["dt"],
                    details={"odd": e["odd"], "ending": frac_str},
                )
            )
        if frac_str in {"0.86", "0.01", "0.07"}:
            signals.append(
                Signal(
                    rule="6",
                    signal="50x",
                    base_time=e["dt"],
                    details={"odd": e["odd"], "ending": frac_str},
                )
            )

    # Rule 1: each pink predicts next pink at +3, +5, +8 minutes
    for e in pinks:
        predicted = [e["dt"] + timedelta(minutes=m) for m in ADDS]
        signals.append(
            Signal(
                rule="1",
                signal="pink",
                base_time=e["dt"],
                predicted_times=predicted,
            )
        )

    # Rule 2: two consecutive pinks appearing in consecutive minutes
    for i in range(len(pinks) - 1):
        a, b = pinks[i], pinks[i + 1]
        if b["minute"] - a["minute"] == 1:
            predicted = [b["dt"] + timedelta(minutes=m) for m in ADDS]
            signals.append(
                Signal(
                    rule="2",
                    signal="pink",
                    base_time=b["dt"],
                    predicted_times=predicted,
                    details={"sequence": [a["dt"], b["dt"]]},
                )
            )

    # Rule 3: three consecutive pinks → use middle pink time
    for i in range(len(pinks) - 2):
        a, b, c = pinks[i], pinks[i + 1], pinks[i + 2]
        if (b["minute"] - a["minute"] == 1) and (c["minute"] - b["minute"] == 1):
            predicted = [b["dt"] + timedelta(minutes=m) for m in ADDS]
            signals.append(
                Signal(
                    rule="3",
                    signal="pink",
                    base_time=b["dt"],
                    predicted_times=predicted,
                    details={"sequence": [a["dt"], b["dt"], c["dt"]]},
                )
            )

    # Rule 4: two pinks with a jumped minute between (gap of exactly 2 minutes)
    for i in range(len(pinks) - 1):
        a, b = pinks[i], pinks[i + 1]
        if b["minute"] - a["minute"] == 2:
            jumped_minute = a["minute"] + 1
            base_dt = datetime.utcfromtimestamp(jumped_minute * 60)
            predicted = [base_dt + timedelta(minutes=m) for m in ADDS]
            signals.append(
                Signal(
                    rule="4",
                    signal="pink",
                    base_time=base_dt,
                    predicted_times=predicted,
                    details={"pinks": [a["dt"], b["dt"]], "jumped_minute": jumped_minute},
                )
            )

    return signals


# ---------- API Routes ----------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.post("/predict", response_model=List[Signal])
def predict(req: PredictRequest):
    return detect_signals(req.events)


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

===#!/usr/bin/env python3
"""
scrape_faculty.py

Day-1 scraper for the UT Austin CS Professor Finder project.

Same technique as find_friday_office_hours.py: use the OpenAI Responses
API with its built-in web_search tool so the model looks up real,
current pages instead of guessing from training data. Instead of
filtering for Friday-afternoon office hours, this script collects the
general profile fields the Day-1 plan needs (name, title, email,
office, profile_url, office_hours) AND adds a "stale" boolean that
flags records whose source page looks like it hasn't been updated
since before 2020 (old syllabi, old "last updated" footers, etc).

SETUP
-----
1. pip install openai
   (optional, only needed for --upsert-db) pip install psycopg2-binary
2. Set your OpenAI key:
     export OPENAI_API_KEY="sk-..."
3. (optional) Set your Supabase/Postgres connection string as an env
   var -- never hardcode the password in this file or commit it:
     export SUPABASE_DB_URL="postgresql://postgres:YOUR-REAL-PASSWORD@db.emygnxnarbrqwyftknrb.supabase.co:5432/postgres"
4. Put professor names in professors.txt (one per line), or edit the
   DEFAULT_PROFESSORS list below.

USAGE
-----
    python scrape_faculty.py
    python scrape_faculty.py --input professors.txt --output-json profs.json
    python scrape_faculty.py --upsert-db                     # also write to Postgres
    python scrape_faculty.py --db-url "$SUPABASE_DB_URL" --upsert-db

NOTES
-----
- "stale" is a heuristic, not a guarantee. The model is asked to
  report any date/year clue it saw on the page (last_updated_hint),
  and this script double-checks that hint for a 4-digit year: if the
  most recent year mentioned is before STALE_YEAR_CUTOFF (2020), the
  record is forced to stale=True regardless of what the model guessed.
  If no year hint is found at all, we fall back to the model's own
  "stale" judgment (default False). Always spot-check a few records.
- Model/tool names occasionally change on OpenAI's side -- if you get
  an error about an unknown tool or model, check
  https://platform.openai.com/docs and update WEB_SEARCH_TOOL / MODEL.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9 fallback
    ZoneInfo = None

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Run: pip install openai")

logger = logging.getLogger("scrape_faculty")

MODEL = "gpt-4.1"                # change if you want a different model
WEB_SEARCH_TOOL = "web_search"   # change if OpenAI renames the tool
STALE_YEAR_CUTOFF = 2020         # source pages hinting at a year before this are "stale"

# ---------------------------------------------------------------------
# Professor lists (carried over from find_friday_office_hours.py)
# ---------------------------------------------------------------------

AI_ML_PROFESSORS = [
    "Peter Stone",
    "Raymond Mooney",
    "Kristen Grauman",
    "Swarat Chaudhuri",
    "Joydeep Biswas",
    "Qiang Liu",
    "Philipp Krahenbuhl",
    "Alex Dimakis",
    "Inderjit Dhillon",
    "Sujay Sanghavi",
    "Risto Miikkulainen",
    "Isil Dillig",
]

THEORY_ALGORITHMS_PROFESSORS = [
    "Scott Aaronson",
    "David Zuckerman",
    "Adam Klivans",
    "Vijaya Ramachandran",
    "Greg Plaxton",
    "Kevin Tian",
    "Brent Waters",
    "David Wu",
    "Eric Price",
]

DISTRIBUTED_SYSTEMS_PROFESSORS = [
    "Vijay Chidambaram",
    "Aditya Akella",
    "Emmett Witchel",
    "Michael Dahlin",
    "Keshav Pingali",
    "Simon Peter",
]

# Specifically requested professors (names corrected to their actual
# UT Austin CS listings):
#   "Alisa Norman"   -> Alison N. Norman
#   "Paarikh"        -> Devangi N. Parikh
#   "Ravishanker"    -> Veena Ravishankar
#   "Scott"          -> Scott Aaronson (already listed under theory)
REQUESTED_PROFESSORS = [
    "Alison N. Norman",
    "Devangi N. Parikh",
    "Veena Ravishankar",
]

DEFAULT_PROFESSORS = sorted(set(
    AI_ML_PROFESSORS
    + THEORY_ALGORITHMS_PROFESSORS
    + DISTRIBUTED_SYSTEMS_PROFESSORS
    + REQUESTED_PROFESSORS
))

SYSTEM_PROMPT = f"""You are a research assistant. Given a professor's name and
their university/department, use web search to find their CURRENT profile
info, ideally from their official university or department page:

- title (e.g. "Professor", "Associate Professor")
- email
- office (building + room, e.g. "GDC 3.816")
- profile_url (their official faculty page)
- office_hours: current posted day(s) of week and time range(s)
- any explicit date/year clue on the page that hints at how recently it
  was updated (a "last updated" footer, a copyright year, a semester like
  "Fall 2018", a syllabus year, etc.) -- put the raw clue text in
  last_updated_hint, or "" if you saw nothing like that
- your own best guess, "stale": true if the page (or the office-hours
  info specifically) looks like it predates {STALE_YEAR_CUTOFF}, else false

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "professor": "<name>",
  "found": true/false,
  "title": "<title or empty string>",
  "email": "<email or empty string>",
  "office": "<office or empty string>",
  "profile_url": "<url or empty string>",
  "office_hours": [
    {{"day": "Monday", "start": "2:00pm", "end": "3:00pm", "location": ""}}
  ],
  "source_url": "<url or empty string>",
  "last_updated_hint": "<raw text clue or empty string>",
  "stale": true/false
}}

If you cannot find a field, leave it as an empty string / empty list, but
still set "found": true if you found the professor's page at all. Only
include office_hours entries you are reasonably confident about."""


def ask_model(client: OpenAI, professor: str) -> dict:
    """Query the model (with web search) for one professor's profile."""
    user_prompt = (
        f"Professor: {professor}\n"
        f"Department: Computer Science, University of Texas at Austin\n"
        "Find their current profile info and office hours."
    )

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": WEB_SEARCH_TOOL}],
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()

    # Model sometimes wraps JSON in ```json fences despite instructions.
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "professor": professor,
            "found": False,
            "title": "",
            "email": "",
            "office": "",
            "profile_url": "",
            "office_hours": [],
            "source_url": "",
            "last_updated_hint": "",
            "stale": False,
            "_raw_response": text,
        }

    data.setdefault("professor", professor)
    return data


def refine_staleness(record: dict) -> dict:
    """
    Cross-check the model's own "stale" guess against any year mentioned
    in last_updated_hint. A concrete year in the hint overrides the
    model's guess; with no hint at all, we keep the model's guess.
    """
    hint = record.get("last_updated_hint") or ""
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", hint)]
    # re.findall with a capturing group inside \b(19|20)\d{2}\b only
    # returns the captured "19"/"20" prefix, so re-extract full years:
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", hint)]

    if years:
        most_recent = max(years)
        record["last_updated_hint_year"] = most_recent
        record["stale"] = most_recent < STALE_YEAR_CUTOFF
    else:
        record["last_updated_hint_year"] = None
        record["stale"] = bool(record.get("stale", False))

    return record


# ---------------------------------------------------------------------
# Optional: push results straight into Supabase/Postgres (Day-1 plan)
#
# This matches the ACTUAL schema you already created in Supabase:
#   professors(id, created_at, name, title, email, office, profile_url,
#              last_updated)
#   office_hours(id, created_at, professor_id, day int8, start_time
#                timestamptz, end_time timestamptz, location, stale bool)
#
# Two things worth knowing about how this maps our JSON onto that schema:
#   - "stale" lives on office_hours, one row per office-hours entry (not
#     one flag per professor), so every entry we insert for a professor
#     gets that professor's page-level stale value copied onto it.
#   - "day" is an int8, so office hours are stored as ISO weekday numbers
#     (Monday=1 ... Sunday=7) via DAY_TO_ISO_WEEKDAY below.
#   - start_time / end_time are timestamptz, so a bare time like "2:00pm"
#     gets attached to an arbitrary anchor date (OFFICE_HOURS_ANCHOR_DATE)
#     in OFFICE_HOURS_TIMEZONE just so Postgres has a full timestamp to
#     store -- only the time-of-day part is meaningful, the date part is
#     a placeholder. Adjust the anchor date/timezone below if you'd
#     rather standardize on something else.
# ---------------------------------------------------------------------

DAY_TO_ISO_WEEKDAY = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
}

OFFICE_HOURS_ANCHOR_DATE = (2000, 1, 1)  # (year, month, day) -- arbitrary
OFFICE_HOURS_TIMEZONE = "America/Chicago"  # UT Austin's timezone


def day_to_int(day_str: str):
    return DAY_TO_ISO_WEEKDAY.get((day_str or "").strip().lower())


def parse_time_to_hour(t: str):
    """Parse a time string like '2:30pm' -> 14.5 (float hour, 24h clock)."""
    t = (t or "").strip().lower().replace(" ", "")
    m = re.match(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", t)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour + minute / 60


def hour_to_timestamp(hour_float):
    """Combine a float hour with the anchor date/timezone into a datetime
    suitable for a timestamptz column. Returns None if hour_float is None."""
    if hour_float is None:
        return None
    hh = int(hour_float)
    mm = int(round((hour_float - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    year, month, day = OFFICE_HOURS_ANCHOR_DATE
    tz = ZoneInfo(OFFICE_HOURS_TIMEZONE) if ZoneInfo else None
    return datetime(year, month, day, hh % 24, mm, tzinfo=tz)


def sanitize_db_url(url: str) -> str:
    """Mask the password in a connection string for safe logging."""
    return re.sub(r"(://[^:/@]+:)[^@]*(@)", r"\1***\2", url or "")


def test_db_connection(db_url: str) -> bool:
    """
    Standalone connectivity check: connects, prints the server version and
    which of the two tables exist plus their columns, then returns.
    Run with `--test-db-connection` to isolate DB problems from the
    scraping part of the pipeline -- if this fails, the scraper never
    even gets a chance to write anything.
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("Missing dependency. Run: pip install psycopg2-binary")
        return False

    logger.info(f"Connecting to {sanitize_db_url(db_url)} ...")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
    except Exception as e:
        logger.error(f"CONNECTION FAILED: {type(e).__name__}: {e}")
        logger.error(
            "Common causes: wrong password, wrong host, project paused in "
            "Supabase, or your network/firewall blocking outbound 5432. "
            "Double check the connection string in the Supabase dashboard "
            "under Project Settings > Database."
        )
        return False

    logger.info("Connected successfully.")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            logger.info(f"Server version: {cur.fetchone()[0]}")

            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('professors', 'office_hours');
            """)
            found_tables = [r[0] for r in cur.fetchall()]
            logger.info(f"Tables found in 'public' schema: {found_tables or 'NONE'}")
            logger.info(len(found_tables))

            for table in ("professors", "office_hours"):
                if table not in found_tables:
                    continue
                cur.execute("""
                    SELECT column_name, data_type FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                """, (table,))
                cols = ", ".join(f"{n}({t})" for n, t in cur.fetchall())
                logger.info(f"{table} columns: {cols}")

            cur.execute("SELECT 1 FROM pg_constraint WHERE conname = 'professors_email_key';")
            logger.info(
                "Unique constraint on professors.email: "
                + ("present" if cur.fetchone() else "MISSING (upserts will fail until ensure_schema adds it)")
            )
    except Exception as e:
        logger.error(f"Connected, but a diagnostic query failed: {type(e).__name__}: {e}")
        conn.close()
        return False

    conn.close()
    logger.info("Connection test passed.")
    return True


def ensure_schema(conn) -> None:
    """
    Creates the tables if they don't exist yet (matching your real
    schema), and additively extends `professors` with a few extra audit
    columns (source_url, found, last_updated_hint, last_updated_hint_year)
    that aren't in your screenshot but are useful to keep. Also makes
    sure `professors.name` has a unique constraint, since the upsert
    below relies on ON CONFLICT (name) -- without it, every upsert
    fails and nothing gets written, silently aborting the whole run.
    Uses autocommit so one failed statement (e.g. the unique constraint,
    if you already have duplicate names) can't roll back the others.
    """
    prev_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        logger.debug("Creating professors table (if not exists) ...")
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS professors (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    name TEXT NOT NULL,
                    title TEXT,
                    email TEXT,
                    office TEXT,
                    profile_url TEXT,
                    last_updated TIMESTAMPTZ DEFAULT now()
                );
            """)
        logger.debug("Creating office_hours table (if not exists) ...")
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS office_hours (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    professor_id BIGINT REFERENCES professors(id) ON DELETE CASCADE,
                    day BIGINT,
                    start_time TIMESTAMPTZ,
                    end_time TIMESTAMPTZ,
                    location TEXT,
                    stale BOOLEAN DEFAULT FALSE
                );
            """)
        logger.debug("Adding audit columns to professors (if missing) ...")
        for stmt in (
            "ALTER TABLE professors ADD COLUMN IF NOT EXISTS source_url TEXT;",
            "ALTER TABLE professors ADD COLUMN IF NOT EXISTS found BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE professors ADD COLUMN IF NOT EXISTS last_updated_hint TEXT;",
            "ALTER TABLE professors ADD COLUMN IF NOT EXISTS last_updated_hint_year INTEGER;",
        ):
            with conn.cursor() as cur:
                cur.execute(stmt)

        # Drop the old name-based constraint if an earlier run of this
        # script added one -- we're keying on email now instead.
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE professors DROP CONSTRAINT IF EXISTS professors_name_key;")

        logger.debug("Checking for unique constraint on professors.email ...")
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_constraint WHERE conname = 'professors_email_key';")
            has_unique = cur.fetchone() is not None
        if not has_unique:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE professors ADD CONSTRAINT professors_email_key UNIQUE (email);"
                    )
                logger.info("Added missing unique constraint on professors.email.")
            except Exception as e:
                logger.error(
                    f"Could not add a unique constraint on professors.email: "
                    f"{type(e).__name__}: {e}. If there are already duplicate "
                    f"(non-null) emails in the table, dedupe them first -- otherwise "
                    f"every ON CONFLICT (email) upsert below will fail with the same error."
                )
        else:
            logger.debug("Unique constraint on professors.email already present.")
    except Exception as e:
        logger.error(f"SCHEMA SETUP FAILED: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        conn.autocommit = prev_autocommit


def upsert_professor_record(conn, record: dict) -> None:
    name = record.get("professor", "<unknown>")
    # Store a missing/blank email as NULL rather than "": Postgres treats
    # every NULL as distinct under a UNIQUE constraint, but two rows with
    # email = '' would collide and fail the upsert.
    email = record.get("email") or None
    if email is None:
        logger.warning(
            f"[{name}] no email found -- professors.email will be NULL. "
            f"Since the unique constraint is on email, this record can't be "
            f"matched/updated by ON CONFLICT next run and will insert as a new row "
            f"if you re-scrape without also having a real email by then."
        )

    with conn.cursor() as cur:
        logger.debug(f"[{name}] Upserting professors row (keyed on email={email!r}) ...")
        cur.execute("""
            INSERT INTO professors
                (name, title, email, office, profile_url, source_url,
                 found, last_updated_hint, last_updated_hint_year, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (email) DO UPDATE SET
                name = EXCLUDED.name,
                title = EXCLUDED.title,
                office = EXCLUDED.office,
                profile_url = EXCLUDED.profile_url,
                source_url = EXCLUDED.source_url,
                found = EXCLUDED.found,
                last_updated_hint = EXCLUDED.last_updated_hint,
                last_updated_hint_year = EXCLUDED.last_updated_hint_year,
                last_updated = now()
            RETURNING id;
        """, (
            record.get("professor", ""),
            record.get("title", ""),
            email,
            record.get("office", ""),
            record.get("profile_url", ""),
            record.get("source_url", ""),
            bool(record.get("found", False)),
            record.get("last_updated_hint", ""),
            record.get("last_updated_hint_year"),
        ))
        professor_id = cur.fetchone()[0]
        logger.debug(f"[{name}] professors.id = {professor_id}")

        cur.execute("DELETE FROM office_hours WHERE professor_id = %s;", (professor_id,))

        page_is_stale = bool(record.get("stale", False))
        inserted = 0
        for entry in record.get("office_hours", []):
            day_int = day_to_int(entry.get("day", ""))
            if day_int is None:
                logger.warning(
                    f"[{name}] unrecognized day '{entry.get('day')}' "
                    f"(expected Monday..Sunday) -- skipping this office-hours row"
                )
                continue
            start_ts = hour_to_timestamp(parse_time_to_hour(entry.get("start", "")))
            end_ts = hour_to_timestamp(parse_time_to_hour(entry.get("end", "")))
            if start_ts is None or end_ts is None:
                logger.warning(
                    f"[{name}] could not parse start/end time "
                    f"({entry.get('start')!r} / {entry.get('end')!r}) -- skipping this row"
                )
                continue
            cur.execute("""
                INSERT INTO office_hours (professor_id, day, start_time, end_time, location, stale)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                professor_id,
                day_int,
                start_ts,
                end_ts,
                entry.get("location", ""),
                page_is_stale,
            ))
            inserted += 1
        logger.debug(f"[{name}] inserted {inserted} office_hours row(s)")
    conn.commit()


def upsert_all(records: list, db_url: str) -> int:
    """Upserts every record; returns the count that failed. A failure on
    one professor rolls back just that professor's transaction and moves
    on, rather than aborting (and silently discarding) the whole batch."""
    try:
        import psycopg2
    except ImportError:
        logger.error("Missing dependency for --upsert-db. Run: pip install psycopg2-binary")
        sys.exit(1)

    logger.info(f"Connecting to {sanitize_db_url(db_url)} ...")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
    except Exception as e:
        logger.error(f"DB CONNECTION FAILED: {type(e).__name__}: {e}")
        logger.error("Run with --test-db-connection for a more detailed connectivity check.")
        raise

    logger.info("Connected. Ensuring schema ...")
    try:
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    logger.info("Schema ready.")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM professors;")
        before_professors = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM office_hours;")
        before_office_hours = cur.fetchone()[0]
    logger.info(
        f"BEFORE this run: professors={before_professors} row(s), "
        f"office_hours={before_office_hours} row(s)"
    )

    failures = 0
    try:
        for record in records:
            name = record.get("professor", "<unknown>")
            try:
                upsert_professor_record(conn, record)
                logger.info(f"[{name}] upserted OK")
            except Exception as e:
                failures += 1
                conn.rollback()
                logger.error(f"[{name}] UPSERT FAILED: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM professors;")
            after_professors = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM office_hours;")
            after_office_hours = cur.fetchone()[0]
        logger.info(
            f"AFTER this run: professors={after_professors} row(s) "
            f"(delta {after_professors - before_professors:+d}), "
            f"office_hours={after_office_hours} row(s) "
            f"(delta {after_office_hours - before_office_hours:+d})"
        )
        conn.close()

    return failures


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Text file with one professor name per line")
    parser.add_argument("--output-json", default="professors.json",
                         help="JSON file to write full results to")
    parser.add_argument("--output-csv", default="professors.csv",
                         help="CSV file to write full results to")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to sleep between API calls (rate-limit friendliness)")
    parser.add_argument("--upsert-db", action="store_true",
                         help="Also upsert results into Postgres/Supabase")
    parser.add_argument("--db-url", default=os.environ.get("SUPABASE_DB_URL", ""),
                         help="Postgres connection string; defaults to $SUPABASE_DB_URL")
    parser.add_argument("--test-db-connection", action="store_true",
                         help="Just check DB connectivity/schema and exit -- no scraping, "
                              "no OPENAI_API_KEY required")
    parser.add_argument("--init-schema", action="store_true",
                         help="Just run schema setup (create tables, add missing columns/"
                              "constraints) and exit -- no scraping, no OPENAI_API_KEY required")
    parser.add_argument("--verbose", "-v", action="store_true",
                         help="Show DEBUG-level logs (exact rows/timestamps being written, etc.)")
    parser.add_argument("--log-file", help="Also write logs to this file (in addition to stdout)")
    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    if args.test_db_connection:
        if not args.db_url:
            logger.error(
                'Set SUPABASE_DB_URL (or pass --db-url), e.g.\n'
                '  export SUPABASE_DB_URL="postgresql://postgres:YOUR-REAL-PASSWORD@'
                'db.emygnxnarbrqwyftknrb.supabase.co:5432/postgres"'
            )
            sys.exit(1)
        ok = test_db_connection(args.db_url)
        sys.exit(0 if ok else 1)

    if args.init_schema:
        if not args.db_url:
            logger.error(
                'Set SUPABASE_DB_URL (or pass --db-url), e.g.\n'
                '  export SUPABASE_DB_URL="postgresql://postgres:YOUR-REAL-PASSWORD@'
                'db.emygnxnarbrqwyftknrb.supabase.co:5432/postgres"'
            )
            sys.exit(1)
        try:
            import psycopg2
        except ImportError:
            logger.error("Missing dependency. Run: pip install psycopg2-binary")
            sys.exit(1)
        logger.info(f"Connecting to {sanitize_db_url(args.db_url)} ...")
        try:
            conn = psycopg2.connect(args.db_url, connect_timeout=10)
        except Exception as e:
            logger.error(f"CONNECTION FAILED: {type(e).__name__}: {e}")
            sys.exit(1)
        try:
            ensure_schema(conn)
        except Exception:
            conn.close()
            sys.exit(1)
        conn.close()
        logger.info("Schema is up to date.")
        sys.exit(0)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment first.")

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            professors = [line.strip() for line in f if line.strip()]
    else:
        professors = DEFAULT_PROFESSORS

    if args.upsert_db and not args.db_url:
        sys.exit(
            "Set SUPABASE_DB_URL (or pass --db-url) before using --upsert-db, e.g.\n"
            '  export SUPABASE_DB_URL="postgresql://postgres:YOUR-REAL-PASSWORD@'
            'db.emygnxnarbrqwyftknrb.supabase.co:5432/postgres"'
        )

    client = OpenAI(api_key=api_key)

    all_results = []

    for i, professor in enumerate(professors, 1):
        logger.info(f"[{i}/{len(professors)}] Looking up {professor} ...")
        try:
            data = ask_model(client, professor)
        except Exception as e:
            logger.error(f"  API call failed for {professor}: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
            data = {
                "professor": professor, "found": False, "title": "", "email": "",
                "office": "", "profile_url": "", "office_hours": [], "source_url": "",
                "last_updated_hint": "", "stale": False,
            }

        data = refine_staleness(data)
        stale_tag = " [STALE]" if data.get("stale") else ""
        logger.info(f"  found={data.get('found')} stale={data.get('stale')}{stale_tag}")

        all_results.append(data)
        time.sleep(args.delay)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # NOTE: this CSV is for human review/auditing only (e.g. eyeballing
    # which pages got flagged stale before you trust them). It is NOT
    # meant to be imported directly into Supabase's table editor -- the
    # column names/types here (plain text days/times, a flat "stale")
    # don't match the real professors/office_hours tables. Use
    # --upsert-db to actually load data into Postgres; it now writes
    # each field with the right type into the right table.
    fieldnames = [
        "professor", "found", "title", "email", "office", "profile_url",
        "source_url", "stale", "last_updated_hint", "last_updated_hint_year",
        "office_hours",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["office_hours"] = json.dumps(r.get("office_hours", []))
            writer.writerow(row)

    stale_count = sum(1 for r in all_results if r.get("stale"))
    logger.info(f"{len(all_results)} professors processed, {stale_count} flagged stale.")
    logger.info(f"Full results saved to: {args.output_json} and {args.output_csv}")

    if args.upsert_db:
        logger.info("Upserting into Postgres/Supabase ...")
        try:
            failures = upsert_all(all_results, args.db_url)
        except Exception as e:
            logger.error(f"Upsert step aborted: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
            sys.exit(1)
        if failures:
            logger.warning(f"Done, but {failures} professor(s) failed to upsert -- see errors above.")
            sys.exit(1)
        logger.info("Done. All records upserted.")


if __name__ == "__main__":
    main()

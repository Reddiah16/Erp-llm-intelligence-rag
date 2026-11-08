"""Repeatable, transactional full-refresh migration into PostgreSQL.

    python -m src.db.migrate              # create objects, load, validate
    python -m src.db.migrate --reset      # drop the erp and stg schemas first
    python -m src.db.migrate --data-only  # keep the objects, reload the rows

The whole run is a single transaction. Staging rows, transformed rows and the
validation queries all happen before one commit, so a failed check leaves the
database exactly as it was found: there is no half-loaded state to notice
later, and no cleanup step to remember.

Data reaches staging through client-side COPY: psycopg streams the raw file
bytes over the connection, so the server never needs to see `data/`, no
superuser grant is required and the files stay where the loader put them. The
byte-level scan during planning found no double quotes, no backslashes, no
carriage returns and no non-ASCII characters in the seven files, which is what
makes `FORMAT csv` with a tab delimiter and `NULL ''` sufficient here.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.data import loader
from src.db import naming, validation
from src.db.config import config_from_env

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_DIR   = ROOT_DIR / "db"

STAGING_SQL   = DB_DIR / "staging.sql"
SCHEMA_SQL    = DB_DIR / "schema.sql"
VIEWS_SQL     = DB_DIR / "views.sql"
TRANSFORM_SQL = DB_DIR / "transform.sql"

COPY_OPTIONS = "WITH (FORMAT csv, DELIMITER E'\\t', NULL '')"

# The raw files, resolved at call time so a test can point the migration at a
# fixture directory the same way it points the loader there.
SOURCE_FILES = loader.required_files


class MigrationError(RuntimeError):
    """A validation gate failed; the transaction was rolled back."""


def read_sql(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise FileNotFoundError(f"{path.name} is empty - the migration cannot continue")
    return text


def drop_schemas(conn) -> None:
    for schema in ("erp", "stg"):
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def create_objects(conn, reset: bool = False) -> List[str]:
    applied = []
    if reset:
        drop_schemas(conn)
        applied.append("dropped schemas erp, stg")
    for path in (STAGING_SQL, SCHEMA_SQL, VIEWS_SQL):
        conn.execute(read_sql(path))
        applied.append(path.name)
    return applied


def assert_staging_matches_source_files(cur) -> None:
    """Fail loudly if db/staging.sql no longer mirrors loader.HEADERS.

    COPY fills columns by position, so a renamed, added or reordered column in
    the loader would otherwise shift every value one slot and still load
    cleanly. This is the check that makes that impossible.
    """
    problems = []
    for raw_name in loader.HEADERS:
        snake    = naming.to_snake(raw_name)
        expected = naming.staging_columns(raw_name)      # snake_case, in file order
        rows = cur.execute(
            "SELECT a.attname FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'stg' AND c.relname = %s "
            "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
            (snake,)).fetchall()
        actual = [r[0] for r in rows]
        if not actual:
            problems.append(f"stg.{snake} does not exist")
        elif actual != expected:
            problems.append(
                f"stg.{snake} columns are {actual} but the loader expects "
                f"{expected} (from loader.HEADERS['{raw_name}'])")
    if problems:
        raise MigrationError("staging schema does not mirror the raw files:\n  - "
                             + "\n  - ".join(problems))


def truncate_staging(cur) -> None:
    for raw_name in loader.HEADERS:
        cur.execute(f"TRUNCATE TABLE {naming.staging_table(raw_name)}")


def copy_file(cur, raw_name: str, path: Path, verbose: bool = False) -> int:
    """Stream one raw TSV into its staging table; returns the row count."""
    table = naming.staging_table(raw_name)
    if not path.exists():
        raise MigrationError(f"{raw_name}: expected file at {path}")
    start = time.perf_counter()
    with cur.copy(f"COPY {table} FROM STDIN {COPY_OPTIONS}") as copy:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                copy.write(chunk)
    rows = cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    if verbose:
        print(f"  COPY {raw_name:22s} -> {table:28s} {rows:>7,} rows "
              f"in {time.perf_counter() - start:.2f}s")
    return rows


def load_staging(conn, files: Dict[str, Path], verbose: bool = True) -> Dict[str, int]:
    cur = conn.cursor()
    truncate_staging(cur)
    assert_staging_matches_source_files(cur)
    return {raw_name: copy_file(cur, raw_name, files[raw_name], verbose)
            for raw_name in loader.HEADERS}


def run_transform(conn) -> None:
    conn.execute(read_sql(TRANSFORM_SQL))


def analyze_erp_tables(conn) -> None:
    """Refresh the statistics the analytics queries plan against.

    A bare ANALYZE also walks the shared catalogs, which a non-superuser may
    not read: it succeeds and fills the server log with "permission denied to
    analyze pg_shdepend" warnings that look like a failing load. Naming the
    seven tables does the same job with nothing to explain afterwards.
    """
    conn.execute("ANALYZE " + ", ".join(f"erp.{table}" for table in naming.ERP_LOAD_ORDER))


def migrate(conn, reset: bool = False, data_only: bool = False,
            validate: bool = True, verbose: bool = True,
            files: Optional[Dict[str, Path]] = None,
            raw_counts: Optional[Dict[str, int]] = None) -> List[validation.Check]:
    """Full refresh inside one transaction. Returns the checks that gated it.

    The caller owns the connection: this function commits or rolls back, and
    never closes it. `with conn:` would have closed it too - psycopg 3 exits a
    connection context by committing *and* disconnecting, which is what a
    script wants and what a caller who plans to keep querying does not.
    """
    # Preflight the dataset before anything touches the database, reusing the
    # loader's own validation so format rules are not written twice.
    raw_counts = raw_counts if raw_counts is not None else loader.validate_dataset()
    files = files if files is not None else SOURCE_FILES()

    started = time.perf_counter()
    try:
        cur = conn.cursor()
        if not data_only:
            for step in create_objects(conn, reset=reset):
                if verbose:
                    print(f"  applied {step}")
        load_staging(conn, files, verbose=verbose)
        run_transform(conn)
        if verbose:
            print("  transform stg -> erp complete")

        if not validate:
            if verbose:
                print("  VALIDATION SKIPPED on request - the gates above are "
                      "what make this load trustworthy")
            conn.commit()
            return []

        checks = validation.run_all(conn, raw_counts)
        bad = validation.failures(checks)
        if bad:
            # Raising here is what rolls the whole load back; the report is
            # attached so the operator sees every failure at once.
            raise MigrationError(
                f"{len(bad)} of {len(checks)} validation checks failed; the "
                f"migration was rolled back and the database is unchanged.\n"
                + validation.format_report(checks))
        analyze_erp_tables(conn)
        conn.commit()
    except BaseException:
        # Every staging row, transformed row and index build below the last
        # commit disappears together; there is no half-loaded state to clean.
        try:
            conn.rollback()
        except Exception as exc:                   # noqa: BLE001 - reported below
            print(f"rollback also failed, the connection may be gone: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    if verbose:
        print(validation.format_report(checks))
        print(f"  committed in {time.perf_counter() - started:.1f}s")
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.db.migrate",
        description="Full-refresh migration of the AdventureWorks files into "
                    "the stg and erp schemas, gated by validation.")
    parser.add_argument("--reset", action="store_true",
                        help="drop the erp and stg schemas before recreating them")
    parser.add_argument("--data-only", action="store_true",
                        help="skip DDL and reload the rows into the existing objects")
    parser.add_argument("--skip-validate", action="store_true",
                        help="load without the validation gate (development only)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    try:
        config = config_from_env()
    except RuntimeError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    import psycopg
    try:
        conn = psycopg.connect(config.conninfo)
    except Exception as exc:                       # noqa: BLE001 - reported below
        # The DSN is never printed; psycopg messages do not contain it.
        print(f"could not connect to PostgreSQL: {type(exc).__name__}: {exc}\n"
              f"target was {config.safe_dsn}", file=sys.stderr)
        return 2

    if verbose:
        print(f"migrating to {config.safe_dsn}")
    try:
        migrate(conn, reset=args.reset, data_only=args.data_only,
                validate=not args.skip_validate, verbose=verbose)
        return 0
    except MigrationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except loader.DatasetError as exc:
        print(f"dataset preflight failed, nothing was loaded:\n{exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

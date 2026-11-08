"""PostgreSQL connection settings, read from the environment only.

Nothing here knows a password. The five POSTGRES_* variables come from the
process environment or from an untracked `.env` file (already covered by
.gitignore), and `.env.example` carries names with placeholder values.

Every string this module produces that could reach a log or an error message
masks the password, because a connection failure that prints its own DSN is
the easiest way for a credential to end up in a terminal transcript, a test
report or a README.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ENV_VARS = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
            "POSTGRES_USER", "POSTGRES_PASSWORD")

DEFAULT_PORT = 5432
DEFAULT_HOST = "localhost"
DEFAULT_DB   = "erp_analytics"
DEFAULT_USER = "erp"


@dataclass(frozen=True)
class PostgresConfig:
    """Where and how to reach the ERP analytics database."""

    host:     str
    port:     int
    dbname:   str
    user:     str
    password: str = field(repr=False)          # excluded from the dataclass repr
    timeout:  int = 10

    @property
    def conninfo(self) -> str:
        """A psycopg connection string. Contains the password: never log it."""
        from psycopg.conninfo import make_conninfo
        return make_conninfo(
            host=host_or_default(self.host), port=self.port, dbname=self.dbname,
            user=self.user, password=self.password, connect_timeout=self.timeout,
        )

    @property
    def safe_dsn(self) -> str:
        """What may be printed: the same coordinates, password redacted."""
        return (f"postgresql://{self.user}@{host_or_default(self.host)}:"
                f"{self.port}/{self.dbname}?password=***redacted***")

    def __repr__(self) -> str:
        return (f"PostgresConfig(host={self.host!r}, port={self.port!r}, "
                f"dbname={self.dbname!r}, user={self.user!r}, password=***)")


def host_or_default(host: str) -> str:
    return host or DEFAULT_HOST


def config_from_env() -> PostgresConfig:
    """Build the configuration, refusing to guess a password.

    Raises RuntimeError naming the missing variable instead of connecting with
    a blank credential and failing somewhere less obvious.
    """
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD is not set. Copy the POSTGRES_* names from "
            ".env.example into .env (gitignored) and fill in the values there; "
            "never put a real password in .env.example, the README or a commit."
        )
    port_raw = os.getenv("POSTGRES_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_raw)
    except ValueError:
        raise RuntimeError(f"POSTGRES_PORT must be a number, got {port_raw!r}")

    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", DEFAULT_HOST),
        port=port,
        dbname=os.getenv("POSTGRES_DB", DEFAULT_DB),
        user=os.getenv("POSTGRES_USER", DEFAULT_USER),
        password=password,
    )


def connect(config: Optional[PostgresConfig] = None):
    """Open a connection with autocommit off, so callers own the transaction."""
    import psycopg
    cfg = config or config_from_env()
    return psycopg.connect(cfg.conninfo)


def server_reachable(config: Optional[PostgresConfig] = None) -> tuple:
    """(reachable, reason) without raising - used by tests to decide a skip.

    Reports which stage failed, but never the credential that was tried.
    """
    try:
        cfg = config or config_from_env()
    except RuntimeError as exc:
        return False, str(exc)
    try:
        import psycopg
    except ImportError:
        return False, "psycopg is not installed in this interpreter"
    try:
        conn = psycopg.connect(cfg.conninfo, connect_timeout=3)
    except Exception as exc:                          # noqa: BLE001 - reported
        return False, f"connect to {cfg.safe_dsn} failed: {type(exc).__name__}: {exc}"
    try:
        row = conn.execute("SELECT current_database(), current_user").fetchone()
        return True, f"connected to {row[0]} as {row[1]}"
    finally:
        conn.rollback()
        conn.close()

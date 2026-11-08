"""Database tests that need no server.

These guard the things that can be wrong before a single row is loaded: the
staging mirror no longer matching the raw file layout, the inventory view
gaining a COALESCE, an analytics query summing order-level money over a joined
line set, transaction control hiding in a SQL file, a credential in
.env.example, or the dependency not being pinned.

Everything here reads files as text, so the suite stays fast and runs on a
machine with no PostgreSQL installed.
"""
import re
from decimal import Decimal
from pathlib import Path

import pytest

from src.data import loader
from src.db import analytics, naming, validation

ROOT_DIR  = Path(__file__).resolve().parents[1]
DB_DIR    = ROOT_DIR / "db"


def sql_text(name: str) -> str:
    return (DB_DIR / f"{name}.sql").read_text(encoding="utf-8")


def sql_code(name: str) -> str:
    """The SQL of one file with its comments removed.

    Prose about the rules belongs in the comments, so a rule check has to read
    the statements only: views.sql explains that COALESCE appears nowhere, and
    that explanation must not trip the check it describes.
    """
    lines = [line.split("--")[0] for line in sql_text(name).splitlines()]
    return "\n".join(lines)


def staging_columns_in_sql(table: str) -> list:
    """Column names of one stg table, in the order CREATE TABLE declares them."""
    marker = f"CREATE TABLE IF NOT EXISTS {table} ("
    text   = sql_text("staging")
    start  = text.index(marker) + len(marker)
    body   = text[start:text.index(");", start)]
    return [line.split()[0] for line in body.splitlines() if line.strip()]


def column_line(text: str, table: str, column: str) -> str:
    """The declaration line for one column of one erp table."""
    marker = f"CREATE TABLE IF NOT EXISTS erp.{table} ("
    start  = text.index(marker) + len(marker)
    body   = text[start:text.index("\n);", start)]
    for line in body.splitlines():
        if line.strip().startswith(f"{column} "):
            return line.strip()
    raise AssertionError(f"erp.{table} has no column {column}")


# ------------------------------------------------------------------- staging

@pytest.mark.parametrize("raw_name", sorted(loader.HEADERS))
def test_staging_sql_mirrors_loader_headers(raw_name):
    """db/staging.sql declares exactly the file's columns, in file order.

    COPY fills by position, so this is the check that stops a renamed or
    reordered loader from shifting every value one slot silently.
    """
    expected = naming.staging_columns(raw_name)
    assert staging_columns_in_sql(naming.staging_table(raw_name)) == expected


def test_staging_is_all_text_and_lossless():
    """Staging stores nothing but strings, and drops nothing."""
    text = sql_code("staging")
    assert "TEXT" in text
    for cast in ("::INTEGER", "::NUMERIC", "::DATE", "::TIMESTAMP", "::UUID", "::BOOLEAN"):
        assert cast not in text
    total_declared = sum(len(staging_columns_in_sql(naming.staging_table(n)))
                         for n in loader.HEADERS)
    assert total_declared == sum(len(c) for c in loader.HEADERS.values())


@pytest.mark.parametrize("raw_name", sorted(loader.HEADERS))
def test_every_source_column_reaches_staging_under_a_snake_name(raw_name):
    for column in loader.HEADERS[raw_name]:
        assert naming.to_snake(column) in staging_columns_in_sql(
            naming.staging_table(raw_name)), column


# ----------------------------------------------------------------- erp schema

def test_all_seven_erp_tables_exist_with_their_approved_names():
    text = sql_text("schema")
    for table in naming.ERP_LOAD_ORDER:
        assert f"CREATE TABLE IF NOT EXISTS erp.{table} (" in text
    assert set(naming.ERP_TABLES.values()) == set(naming.ERP_LOAD_ORDER)


def test_primary_keys_are_declared_on_every_table():
    text = sql_text("schema")
    for pk in ("PRIMARY KEY",):
        assert text.count(pk) == 7
    assert "PRIMARY KEY (sales_order_id, sales_order_detail_id)" in text
    assert "PRIMARY KEY (purchase_order_id, purchase_order_detail_id)" in text
    assert "PRIMARY KEY (product_id, location_id)" in text


def test_foreign_keys_only_point_at_loaded_parents():
    """Six FKs, and deliberately none toward the fifteen un-loaded tables."""
    text = sql_text("schema")
    targets = re.findall(r"REFERENCES erp\.(\w+)\(", text)
    assert sorted(set(targets)) == ["customer", "product", "purchase_order",
                                    "sales_order"]
    # A foreign key toward a table that is never loaded cannot be enforced, and
    # a constraint that can never fail is worse than no constraint at all.
    unenforceable = [("purchase_order", "vendor_id"),
                     ("purchase_order", "employee_id"),
                     ("purchase_order", "ship_method_id"),
                     ("customer", "territory_id"),
                     ("sales_order", "ship_method_id"),
                     ("sales_order", "bill_to_address_id"),
                     ("sales_order", "currency_rate_id"),
                     ("sales_order_line", "special_offer_id"),
                     ("product_inventory", "location_id"),
                     ("product", "product_subcategory_id")]
    for table, column in unenforceable:
        assert "REFERENCES" not in column_line(text, table, column), (table, column)


def test_money_columns_keep_the_published_precision():
    text = sql_text("schema")
    assert "line_total               NUMERIC(19,6)" in text
    for table, column in [("sales_order", "total_due"), ("sales_order", "sub_total"),
                          ("sales_order", "tax_amt"), ("sales_order", "freight"),
                          ("purchase_order", "total_due"), ("sales_order_line", "unit_price"),
                          ("product", "list_price"), ("product", "standard_cost")]:
        assert "NUMERIC(19,4)" in column_line(text, table, column)
    # Two-decimal money would silently reproduce the CSV pipeline's rounding.
    assert "NUMERIC(19,2)" not in text and "NUMERIC(10,2)" not in text


def test_quantity_is_not_null_so_absence_stays_separate_from_zero():
    text = sql_text("schema")
    assert "NOT NULL" in column_line(text, "product_inventory", "quantity")
    assert "DEFAULT" not in column_line(text, "product_inventory", "quantity")


def test_arithmetic_identities_that_were_verified_are_enforced():
    text = sql_text("schema")
    assert "CHECK (total_due = sub_total + tax_amt + freight)" in text
    assert "CHECK (stocked_qty = received_qty - rejected_qty)" in text
    assert text.count("total_due = sub_total + tax_amt + freight") == 2


def test_indexes_are_the_essentials_and_no_more():
    text = sql_text("schema")
    indexes = re.findall(r"CREATE INDEX IF NOT EXISTS (\w+)", text)
    assert len(indexes) == 6
    assert {"sales_order_order_date_idx", "sales_order_customer_idx",
            "sales_order_line_product_idx", "purchase_order_status_idx",
            "purchase_order_vendor_idx",
            "purchase_order_line_product_idx"} == set(indexes)


# ---------------------------------------------------------------------- view

def test_stock_position_view_has_no_coalesce():
    """The one edit that would quietly invent 72 stock-outs is refused here.

    A call is matched, not the word: the view's own COMMENT tells the reader
    never to coalesce, and that sentence is the rule being enforced.
    """
    text = sql_code("views").lower()
    assert "coalesce(" not in text
    assert "CREATE OR REPLACE VIEW erp.stock_position" in sql_code("views")
    assert "never coalesce it to zero" in sql_text("views").lower()
    assert "left join" in text and "erp.product p" in text


def test_stock_position_flags_test_for_a_record_explicitly():
    text = sql_code("views")
    assert text.count("r.product_id IS NOT NULL") == 3      # 1 projection + 2 flags
    assert "has_inventory_record" in text
    assert "SUM(quantity)" in text                           # aggregated before joining


def test_stock_position_is_driven_by_product():
    """504 rows come from product, not from the 1,069 inventory rows."""
    text = sql_code("views")
    assert "FROM erp.product p" in text
    # Inventory is aggregated before it is joined, so a product with six
    # locations still produces one row here.
    assert text.index("GROUP  BY product_id") < text.index("FROM erp.product p")


# ---------------------------------------------------------------- transform

def test_transform_loads_every_table_from_staging():
    text = sql_text("transform")
    for table in naming.ERP_LOAD_ORDER:
        assert f"INSERT INTO erp.{table} (" in text
    assert text.count("INSERT INTO") == 7
    assert "TRUNCATE TABLE" in text


def test_transform_casts_instead_of_relying_on_implicit_conversion():
    text = sql_code("transform")
    assert "NULLIF(" in text
    assert "::TIMESTAMP::DATE" in text       # business dates, midnight measured
    assert "::UUID" in text
    assert "::NUMERIC(19,6)" in text         # the six-decimal line figure
    assert "ROUND(" not in text.upper()      # nothing is rounded on the way in


@pytest.mark.parametrize("name", ["staging", "schema", "views", "transform",
                                   "provision", "provision_interactive"])
def test_sql_files_contain_no_transaction_control(name):
    """migrate.py owns the transaction, so a commit here would defeat the gates.

    provision.sql belongs in the list even though psql runs it: the CREATE
    DATABASE in it cannot run inside a transaction block at all, so a BEGIN
    there would not merely skip the gates, it would fail the script.
    """
    text = sql_code(name).upper()
    for statement in ("BEGIN", "COMMIT", "ROLLBACK"):
        assert not re.search(rf"^\s*{statement}\s*;", text), f"{name}.sql has {statement}"


# --------------------------------------------------------------------- gate

def test_row_count_expectations_match_the_loader_reference():
    """One source of truth for the published row counts, checked two ways."""
    by_table = {naming.ERP_TABLES[name]: count
                for name, count in loader.REFERENCE_ROW_COUNTS.items()}
    assert validation.EXPECTED_ERP_ROW_COUNTS == by_table


def test_approved_inventory_figures_are_unchanged():
    assert validation.EXPECTED_STOCK_POSITION == {
        "products": 504, "recorded": 432, "unrecorded": 72,
        "below_safety": 69, "needs_reorder": 7, "confirmed_zero": 4}
    assert sum(validation.EXPECTED_PURCHASE_ORDER_STATUS.values()) == \
        loader.REFERENCE_ROW_COUNTS["PurchaseOrderHeader"]
    assert sum(validation.EXPECTED_SALES_ORDER_STATUS.values()) == \
        loader.REFERENCE_ROW_COUNTS["SalesOrderHeader"]


def test_every_validation_check_is_wired_into_the_gate():
    assert validation.check_row_counts not in validation.ALL_CHECKS  # called by run_all
    assert len(validation.ALL_CHECKS) == 10
    assert validation.check_stock_position in validation.ALL_CHECKS
    assert validation.check_missing_inventory_is_never_zero in validation.ALL_CHECKS
    assert validation.check_no_duplicated_order_revenue in validation.ALL_CHECKS
    assert validation.check_totals_match_source_files in validation.ALL_CHECKS
    assert validation.check_precision_is_preserved in validation.ALL_CHECKS


def test_money_oracle_covers_both_grains_and_the_files_they_come_from():
    tables = {re.search(r"FROM erp\.(\w+)", sql).group(1)
              for sql, _, _ in validation.MONEY_ORACLE}
    assert tables == {"sales_order", "sales_order_line", "purchase_order",
                      "purchase_order_line"}
    # The oracle names source columns in their staging spelling, which is the
    # spelling validation.py reads the raw files with.
    assert all(column in naming.staging_columns(raw_name)
               for _, raw_name, column in validation.MONEY_ORACLE)


def test_check_dataclass_passes_only_on_equality_and_reports_failures():
    good = validation.Check("a", 1, 1)
    bad  = validation.Check("b", 1, 2, "expected one")
    assert good.passed and not bad.passed
    assert validation.failures([good, bad]) == [bad]
    report = validation.format_report([good, bad])
    assert "[PASS] a" in report and "[FAIL] b" in report
    assert "1/2 checks passed" in report and "rolled back" in report
    assert "expected one" in str(bad)


# ------------------------------------------------------------------ analytics

def test_all_eight_questions_are_answered_and_named():
    assert len(analytics.QUESTIONS) == 8
    assert set(analytics.QUESTIONS) == set(analytics.TITLES)
    for number, question in enumerate(analytics.QUESTIONS, start=1):
        assert analytics.TITLES[question].startswith(f"{number}.")


def test_total_revenue_query_sums_each_grain_from_its_own_table():
    """No join, because a join is what multiplies an order-level amount."""
    sql = analytics.TOTAL_SALES_REVENUE_SQL.upper()
    assert "JOIN" not in sql
    assert "SUM(TOTAL_DUE)" in sql and "SUM(LINE_TOTAL)" in sql
    assert "FROM erp.sales_order)" in analytics.TOTAL_SALES_REVENUE_SQL
    assert "FROM erp.sales_order_line)" in analytics.TOTAL_SALES_REVENUE_SQL


def test_product_revenue_is_a_line_grain_figure():
    sql = analytics.TOP_PRODUCTS_BY_REVENUE_SQL
    assert "sum(l.line_total)" in sql
    assert "total_due" not in sql
    assert "GROUP  BY p.product_id" in sql


def test_customer_revenue_is_an_order_grain_figure():
    sql = analytics.TOP_CUSTOMERS_BY_REVENUE_SQL.upper()
    assert "SALES_ORDER_LINE" not in sql
    assert "SUM(O.TOTAL_DUE)" in sql
    assert "GROUP  BY C.CUSTOMER_ID" in sql


def test_monthly_trends_aggregate_each_grain_before_joining():
    sql  = analytics.MONTHLY_SALES_TRENDS_SQL
    text = sql.upper()
    assert "WITH ORDERS AS" in text and "LINES AS" in text
    orders_cte = text.split("LINES AS")[0]
    lines_cte  = text.split("LINES AS")[1].split("SELECT O.MONTH")[0]
    assert "JOIN" not in orders_cte            # order grain straight off the header
    assert "SUM(TOTAL_DUE)" not in lines_cte   # never re-summed over joined lines
    assert "SUM(L.LINE_TOTAL)" in lines_cte


def test_pending_purchase_order_queries_work_at_order_grain():
    for sql in (analytics.PENDING_PURCHASE_ORDER_COUNT_SQL,
                analytics.PENDING_PURCHASE_ORDER_VALUE_SQL):
        assert "erp.purchase_order" in sql and "erp.purchase_order_line" not in sql
        assert "order_status = 1" in sql


def test_stock_alert_queries_read_the_view():
    for sql in (analytics.PRODUCTS_BELOW_SAFETY_STOCK_SQL,
                analytics.PRODUCTS_NEEDING_REORDER_SQL):
        assert "FROM   erp.stock_position" in sql
        assert "product_inventory" not in sql
    assert "WHERE  below_safety" in analytics.PRODUCTS_BELOW_SAFETY_STOCK_SQL
    assert "WHERE  needs_reorder" in analytics.PRODUCTS_NEEDING_REORDER_SQL


def test_analytics_exposes_one_function_per_question():
    for question in analytics.QUESTIONS:
        assert callable(getattr(analytics, question)), question
    assert "limit" in analytics.top_products_by_revenue.__code__.co_varnames


def test_report_prints_a_scalar_a_single_row_and_a_table():
    """The eight answers are three different shapes; none may break the report."""
    text = analytics.format_results({
        "total_sales_revenue": {"orders": 2, "order_total_due": Decimal("9.5000")},
        "top_products_by_revenue": [{"product_id": 7, "revenue": Decimal("4.200000")}],
        "top_customers_by_revenue": [],
        "products_below_safety_stock": [{"product_id": i} for i in range(60)],
        "products_needing_reorder": [],
        "pending_purchase_order_count": 225,
        "pending_purchase_order_value": {"pending_total_due": Decimal("1.2345")},
        "monthly_sales_trends": [{"month": "2024-01"}],
    })
    assert "9.5000" in text                      # four stored decimals survive
    assert "4.200000" in text                   # six on the line grain
    assert "6. Pending purchase order count\n  225" in text
    assert "10 more rows" in text
    assert "7. Total value of pending purchase orders" in text


def test_report_can_print_one_answer_without_the_other_seven():
    """`--question` narrows the results, so the report has to follow them.

    Iterating the question list instead of the results raised KeyError on the
    seven answers that were never asked for - a shape no all-eight report ever
    shows, because the dict handed to it is complete there.
    """
    text = analytics.format_results({"pending_purchase_order_count": 225})
    assert text == "6. Pending purchase order count\n  225"
    assert "1. Total sales revenue" not in text


def test_report_refuses_an_unknown_question_name():
    with pytest.raises(ValueError, match="not a known question"):
        analytics.format_results({"made_up_question": 1})


# ----------------------------------------------------------------- settings

def test_requirements_pin_the_binary_driver():
    for name in ("requirements.api.txt", "requirements.dev.txt"):
        text = (ROOT_DIR / name).read_text(encoding="utf-8")
        assert "psycopg[binary]==3.2.9" in text, name


def test_env_example_lists_names_and_no_secret():
    text = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    for var in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD"):
        assert f"{var}=" in text, var
    for line in text.splitlines():
        if line.startswith("POSTGRES_PASSWORD="):
            assert line.split("=", 1)[1] == "change-me"
    assert ".env" in (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")


def test_provision_sql_carries_no_password_and_asks_for_no_privilege():
    """The one SQL file a superuser runs must not itself want privileges.

    A literal password here would be committed, and a SUPERUSER or
    pg_read_server_files grant would be a standing invitation to read any file
    on the server - the migration needs neither, because COPY FROM STDIN
    streams the data over the connection.
    """
    text = sql_text("provision")
    code = sql_code("provision")
    assert r"\getenv erp_password ERP_DB_PASSWORD" in text
    assert not re.search(r"PASSWORD\s+'", code), "a literal password in a committed file"
    assert r"PASSWORD %L" in code                      # quoted by format(), never concatenated
    assert r"\if :pw_blank" in text
    # Ownership of the database is the whole grant; anything else is a privilege
    # the migration does not need and should not be discovered to rely on. The
    # role statements are read line by line, because the report at the foot of
    # the file selects rolcreaterole and a statement-spanning match would read
    # that column name as a demand for the attribute.
    assert "GRANT" not in code.upper()
    role_lines = [line.strip() for line in code.splitlines()
                  if re.search(r"(CREATE|ALTER) ROLE", line, re.I)]
    assert len(role_lines) == 2, role_lines
    for line in role_lines:
        for too_much in ("SUPERUSER", "CREATEDB", "CREATEROLE", "BYPASSRLS"):
            assert too_much not in line.upper(), (line, too_much)


@pytest.mark.parametrize("name", ["provision", "provision_interactive"])
def test_provisioning_is_idempotent_and_names_the_settings_the_app_expects(name):
    r"""Re-running either script on a provisioned server must change nothing.

    Each of the four statements is a conditional \gexec, and the role and
    database they build are the same ones config.py and .env.example advertise -
    a mismatch there would turn setup into a confusing privilege error.
    """
    from src.db import config
    text = sql_code(name)                       # comments out: the header explains \gexec
    assert text.count("\\gexec") == 4
    assert "WHERE NOT EXISTS (SELECT 1 FROM pg_roles" in text
    assert "WHERE NOT EXISTS (SELECT 1 FROM pg_database" in text
    assert "WHERE NOT (SELECT rolcanlogin" in text
    assert "CREATE ROLE erp" in text and "CREATE DATABASE erp_analytics" in text
    assert config.DEFAULT_USER == "erp"
    assert config.DEFAULT_DB == "erp_analytics"
    env = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    assert "POSTGRES_USER=erp" in env and "POSTGRES_DB=erp_analytics" in env


@pytest.mark.parametrize("name", ["provision", "provision_interactive"])
def test_no_provisioning_script_holds_a_secret_or_asks_for_a_privilege(name):
    r"""Either way in, the least privilege is the same: own one database.

    A literal password in a committed file is a secret in the repository, and
    SUPERUSER or pg_read_server_files would be a standing invitation to read
    any file on the server. The migration needs neither, because COPY FROM
    STDIN streams the data over the connection.

    How the two scripts supply the password differs on purpose - provision.sql
    names it in a CREATE ROLE statement through format()'s %L, this check only
    forbids a quoted literal - so the \password-only rule is asserted in
    test_provision_interactive_leaves_the_password_to_psql.
    """
    code = sql_code(name)
    assert not re.search(r"PASSWORD\s+'", code), "a literal password in a committed file"
    assert "GRANT" not in code.upper()
    assert "pg_read_server_files" not in code
    # The role statements are read line by line, because the report at the foot
    # of each file selects rolcreaterole and a statement-spanning match would
    # read that column name as a demand for the attribute.
    role_lines = [line.strip() for line in code.splitlines()
                  if re.search(r"(CREATE|ALTER) ROLE", line, re.I)]
    assert role_lines, name
    for line in role_lines:
        for too_much in ("SUPERUSER", "CREATEDB", "CREATEROLE", "BYPASSRLS"):
            assert too_much not in line.upper(), (name, line, too_much)


def test_provision_interactive_leaves_the_password_to_psql():
    r"""The point of the interactive variant: no value the client can echo.

    provision.sql needs ERP_DB_PASSWORD because it puts the secret in a CREATE
    ROLE statement. Here the role is created with no verifier at all and
    \password supplies one, so neither the environment nor the query text nor
    the shell history ever holds the secret.
    """
    text = sql_text("provision_interactive")
    code = sql_code("provision_interactive")
    # Comments may name ERP_DB_PASSWORD to explain what this file avoids; the
    # statements must not use it.
    assert "\\getenv" not in code and "ERP_DB_PASSWORD" not in code
    assert code.count("\\password erp") == 1
    # The reason the file exists has to be readable by someone who finds two
    # provisioning scripts and has to pick one.
    assert "provision.sql" in text and "\\password" in text
    # No DDL statement carries a password clause - only \password does.
    for line in code.splitlines():
        if re.search(r"^\s*(SELECT ')?(CREATE|ALTER) (ROLE|DATABASE)", line, re.I):
            assert "PASSWORD" not in line.upper(), line


def test_provision_interactive_prompts_only_when_there_is_no_verifier():
    r"""Re-running setup must not rotate the password .env still holds.

    The \if condition comes from one aggregate row, so a missing role and a
    passwordless role both take the prompt branch and a provisioned role takes
    the echo branch. A zero-row condition would read as true and reset a
    working password.
    """
    code = sql_code("provision_interactive")
    assert "coalesce(bool_or(rolpassword IS NULL), true) AS needs_password" in code
    assert code.index("\\gset pw_") < code.index("\\if :pw_needs_password")
    branch = code.split("\\if :pw_needs_password", 1)[1].split("\\endif", 1)[0]
    assert "\\password erp" in branch.split("\\else", 1)[0], "prompt must be in the then branch"
    assert "\\password" not in branch.split("\\else", 1)[1]
    assert "coalesce(bool_or(rolpassword IS NOT NULL), false) AS has_verifier" in code
    assert "\\if :ok_has_verifier" in code


def test_provision_interactive_never_prints_the_password_verifier():
    """rolpassword is a credential store, not a report column.

    Printing it would put a crackable hash in a terminal transcript. Every use
    of the column is a test on it, and the report describes the type instead.
    """
    code = sql_code("provision_interactive")
    following = [m.group(1) for m in re.finditer(r"rolpassword\s+(\S+)", code)]
    assert following and set(following) <= {"IS", "LIKE"}, following
    assert "AS verifier_type" in code and "LIKE 'SCRAM-SHA-256$%'" in code
    assert "md5" in code, "a weak verifier should be named, not printed"


@pytest.mark.parametrize("name", ["provision", "provision_interactive"])
def test_provisioning_refusals_fail_loudly_instead_of_quitting_quietly(name):
    r"""A setup that refuses must not look like a setup that worked.

    Measured on psql 18: \quit cannot carry an exit code, so it always leaves
    status 0 - a CI step or a follow-on command would go straight to the
    migration after a run that created nothing. Each refusal therefore raises,
    and ON_ERROR_STOP is set inside the file rather than left to whichever
    command line invoked it, so the raise actually stops the run.
    """
    code = sql_code(name)
    assert r"\set ON_ERROR_STOP on" in code
    assert "\\quit" not in code
    assert code.count("RAISE EXCEPTION") == code.count("provisioning refused")
    assert code.count("RAISE EXCEPTION") >= 1, name
    # A refusal must sit in the branch that detects the problem, not after
    # \endif, where it would fire on every run.
    for block in re.findall(r"\\else(.*?)\\endif", code, re.S):
        if "RAISE EXCEPTION" in block:
            assert "\\echo ERROR" in block


def test_config_never_prints_the_password():
    from src.db.config import PostgresConfig
    cfg = PostgresConfig(host="localhost", port=5432, dbname="erp_analytics",
                         user="erp", password="hunter2")
    for rendered in (repr(cfg), str(cfg), cfg.safe_dsn):
        assert "hunter2" not in rendered
    assert "password=***redacted***" in cfg.safe_dsn
    assert "hunter2" in cfg.conninfo            # the one string that must stay private


def test_config_refuses_to_guess_a_missing_password(monkeypatch):
    from src.db.config import config_from_env
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD is not set"):
        config_from_env()


def test_config_reads_coordinates_from_the_environment(monkeypatch):
    from src.db.config import config_from_env
    monkeypatch.setenv("POSTGRES_PASSWORD", "from-test")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5544")
    monkeypatch.setenv("POSTGRES_DB", "other_db")
    monkeypatch.setenv("POSTGRES_USER", "reader")
    cfg = config_from_env()
    assert (cfg.host, cfg.port, cfg.dbname, cfg.user) == ("db.internal", 5544, "other_db", "reader")


def test_config_rejects_a_non_numeric_port(monkeypatch):
    from src.db.config import config_from_env
    monkeypatch.setenv("POSTGRES_PASSWORD", "from-test")
    monkeypatch.setenv("POSTGRES_PORT", "5432a")
    with pytest.raises(RuntimeError, match="POSTGRES_PORT must be a number"):
        config_from_env()


def test_migrate_exposes_the_documented_cli_switches(capsys):
    from src.db import migrate
    with pytest.raises(SystemExit) as exit_code:
        migrate.main(["--help"])
    assert exit_code.value.code == 0
    usage = capsys.readouterr().out
    for flag in ("--reset", "--data-only", "--skip-validate", "--quiet"):
        assert flag in usage, flag
    assert "Full-refresh migration" in usage


def test_migrate_command_uses_client_side_copy_only():
    """Server-side COPY would need the files to be readable by the postmaster."""
    import inspect
    from src.db import migrate
    assert "FROM STDIN" in inspect.getsource(migrate.copy_file)
    assert "FORMAT csv" in migrate.COPY_OPTIONS and "NULL ''" in migrate.COPY_OPTIONS
    assert r"DELIMITER E'\t'" in migrate.COPY_OPTIONS
    assert "COPY" not in sql_code("transform")


def test_copy_writer_method_exists_in_the_pinned_driver():
    """psycopg 3.1 called it write_data(); 3.2 renamed it to write().

    This is the one place a pinned dependency's API is checked rather than
    assumed, because the rename fails the whole migration at the first file.
    """
    import inspect
    from psycopg.copy import Copy

    from src.db import migrate
    used = {name for name in ("write", "write_row", "end", "set_types")
            if f"copy.{name}(" in inspect.getsource(migrate.copy_file)}
    assert used, "copy_file() no longer writes to the COPY object"
    for name in used:
        assert hasattr(Copy, name), f"psycopg {Copy.__module__} has no {name}()"


def test_db_code_only_uses_apis_psycopg_3_has():
    """`.mappings()` is SQLAlchemy; psycopg 3 has no such method.

    Both bugs this guards against are invisible until a query actually runs,
    so the source is searched instead.
    """
    offenders = {}
    for path in sorted((ROOT_DIR / "src" / "db").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for API in (".mappings()", ".row_mapping(", "write_data(", "execute_values("):
            if API in text:
                offenders.setdefault(path.name, []).append(API)
    assert offenders == {}, offenders


def test_migrate_loads_exactly_the_seven_raw_files_the_loader_requires():
    from src.db import migrate
    files = migrate.SOURCE_FILES()
    assert set(files) == set(loader.HEADERS)
    for name, path in files.items():
        assert path.name == f"{name}.csv", name
        assert path.parts[-2] in ("sd", "mm"), name


def test_migration_error_is_raised_before_a_commit_is_reached():
    """The gate has to raise, otherwise a bad load would be committed."""
    from src.db import migrate
    assert issubclass(migrate.MigrationError, RuntimeError)
    text = migrate.migrate.__doc__ or ""
    assert "transaction" in text or "transaction" in migrate.__doc__


def test_migrate_owns_the_transaction_but_not_the_connection():
    """psycopg 3 closes a connection when its `with conn:` block exits.

    A migration written that way commits correctly and still destroys the
    handle the caller passed in, so the commit and the rollback are written
    out rather than delegated to a context manager, and disconnecting is left
    to the command-line entry point that opened the connection. The source is
    read as a syntax tree because the function's own docstring has to explain
    the trap, and a plain substring search would find that explanation.
    """
    import ast
    import inspect

    from src.db import migrate

    def methods_called_on(function, object_name):
        tree = ast.parse(inspect.getsource(function))
        return [node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func.value) == object_name]

    used = methods_called_on(migrate.migrate, "conn")
    assert "close" not in used, "the migration must not disconnect its caller"
    assert used.count("commit") == 2, used        # the load, and --skip-validate
    assert "rollback" in used, "a failed gate has to undo the whole load"
    tree = ast.parse(inspect.getsource(migrate.migrate))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    assert "close" in methods_called_on(migrate.main, "conn")


def test_statistics_refresh_names_the_erp_tables_not_the_whole_cluster():
    """A bare ANALYZE works as a superuser and warns as the `erp` role.

    Measured on a live server: as the non-superuser role it succeeded and
    logged "permission denied to analyze pg_shdepend" for eight shared
    catalogs, which reads to an operator like a load that failed. Naming the
    seven tables collects the statistics that the questions actually plan
    against, and leaves the log clean.
    """
    from src.db import migrate

    class RecordingConnection:
        def __init__(self):
            self.statements = []

        def execute(self, sql, params=None):
            self.statements.append(sql)

    conn = RecordingConnection()
    migrate.analyze_erp_tables(conn)
    assert len(conn.statements) == 1
    sql = conn.statements[0]
    assert sql.upper().startswith("ANALYZE ")
    for table in naming.ERP_LOAD_ORDER:
        assert f"erp.{table}" in sql, table
    assert sql.count("erp.") == len(naming.ERP_LOAD_ORDER)
    for not_mine in ("pg_", "stg.", "VACUUM"):
        assert not_mine not in sql, not_mine


# ------------------------------------------------- what the casts demand of the
# raw files
#
# The migration is one transaction, so a single value PostgreSQL cannot cast,
# or a single duplicate under a UNIQUE constraint, rolls the whole load back.
# These tests find that out from the files first, without needing a server, so
# a failure names the column instead of reporting an aborted transaction.

# The columns erp declares NOT NULL, per raw file, in staging spelling. An
# empty field becomes NULLIF(x, '') -> NULL -> a constraint violation.
REQUIRED_SOURCE_COLUMNS = {
    "SalesOrderHeader": [
        "sales_order_id", "revision_number", "order_date", "due_date", "status",
        "online_order_flag", "sales_order_number", "customer_id", "territory_id",
        "bill_to_address_id", "ship_to_address_id", "ship_method_id",
        "sub_total", "tax_amt", "freight", "total_due", "rowguid",
        "modified_date"],
    "SalesOrderDetail": [
        "sales_order_id", "sales_order_detail_id", "product_id",
        "special_offer_id", "order_qty", "unit_price", "unit_price_discount",
        "line_total", "rowguid", "modified_date"],
    "Customer": [
        "customer_id", "territory_id", "account_number", "rowguid",
        "modified_date"],
    "Product": [
        "product_id", "name", "product_number", "make_flag",
        "finished_goods_flag", "safety_stock_level", "reorder_point",
        "standard_cost", "list_price", "days_to_manufacture",
        "sell_start_date", "rowguid", "modified_date"],
    "ProductInventory": [
        "product_id", "location_id", "bin", "quantity", "rowguid",
        "modified_date"],
    "PurchaseOrderHeader": [
        "purchase_order_id", "revision_number", "status", "employee_id",
        "vendor_id", "ship_method_id", "order_date", "sub_total", "tax_amt",
        "freight", "total_due", "modified_date"],
    "PurchaseOrderDetail": [
        "purchase_order_id", "purchase_order_detail_id", "due_date",
        "order_qty", "product_id", "unit_price", "line_total", "received_qty",
        "rejected_qty", "stocked_qty", "modified_date"],
}

# Only these may be read as BOOLEAN, so "True" or "t" anywhere else is a bug.
BOOLEAN_SOURCE_COLUMNS = {
    "SalesOrderHeader": ["online_order_flag"],
    "Product":          ["make_flag", "finished_goods_flag"],
}

# One row per key, or the primary key or a UNIQUE constraint rejects the load.
UNIQUE_SOURCE_KEYS = {
    "SalesOrderHeader":    [["sales_order_id"], ["sales_order_number"], ["rowguid"]],
    "SalesOrderDetail":    [["sales_order_id", "sales_order_detail_id"], ["rowguid"]],
    "Customer":            [["customer_id"], ["account_number"], ["rowguid"]],
    "Product":             [["product_id"], ["product_number"], ["rowguid"]],
    "ProductInventory":    [["product_id", "location_id"], ["rowguid"]],
    "PurchaseOrderHeader": [["purchase_order_id"]],
    "PurchaseOrderDetail": [["purchase_order_id", "purchase_order_detail_id"]],
}

requires_dataset = pytest.mark.skipif(
    not all(path.exists() for path in loader.required_files().values()),
    reason="the AdventureWorks raw dataset is not present in data/")


def scan_columns(name: str, columns: list, keep_distinct: int = 0):
    """One pass over a raw file: row count, empty values, sampled distinct text."""
    positions = [naming.staging_columns(name).index(column) for column in columns]
    empties   = dict.fromkeys(columns, 0)
    distinct  = {column: set() for column in columns}
    rows = 0
    with loader.required_files()[name].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            rows += 1
            for column, index in zip(columns, positions):
                value = fields[index]
                if not value:
                    empties[column] += 1
                elif keep_distinct and len(distinct[column]) < keep_distinct:
                    distinct[column].add(value)
    return rows, empties, distinct


@requires_dataset
@pytest.mark.parametrize("name", sorted(REQUIRED_SOURCE_COLUMNS))
def test_not_null_source_columns_are_never_empty(name):
    rows, empties, _ = scan_columns(name, REQUIRED_SOURCE_COLUMNS[name])
    assert rows > 0
    assert {c: n for c, n in empties.items() if n} == {}, \
        f"{name}.csv has empty values in a NOT NULL column"


@requires_dataset
@pytest.mark.parametrize("name", sorted(BOOLEAN_SOURCE_COLUMNS))
def test_boolean_source_columns_only_hold_zero_and_one(name):
    for column in BOOLEAN_SOURCE_COLUMNS[name]:
        _, _, distinct = scan_columns(name, [column], keep_distinct=12)
        assert distinct[column] <= {"0", "1"}, (name, column, distinct[column])


@requires_dataset
@pytest.mark.parametrize("name", sorted(UNIQUE_SOURCE_KEYS))
def test_primary_key_and_unique_columns_have_no_repeated_values(name):
    positions = [[naming.staging_columns(name).index(column)
                  for column in group] for group in UNIQUE_SOURCE_KEYS[name]]
    seen, repeats = [set() for _ in positions], []
    with loader.required_files()[name].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            for group, indexes, bucket in zip(UNIQUE_SOURCE_KEYS[name], positions, seen):
                key = tuple(fields[i] for i in indexes)
                if key in bucket:
                    repeats.append((group, key))
                else:
                    bucket.add(key)
    assert not repeats, repeats[:5]


INTEGER_CAST = re.compile(r"NULLIF\((\w+), ''\)::(INTEGER|SMALLINT)")


def transform_integer_casts():
    """{stg table: [(column, sql type), ...]} straight out of db/transform.sql."""
    casts = {}
    for block in sql_code("transform").split("INSERT INTO")[1:]:
        source = re.search(r"FROM (stg\.\w+)", block)
        assert source, "an INSERT without a stg source table"
        casts[source.group(1)] = INTEGER_CAST.findall(block)
    return casts


@requires_dataset
def test_the_whole_number_casts_are_justified_by_the_source_text():
    """Every ::INTEGER and ::SMALLINT in the transform is checked against the text.

    A value published as "3.00" is a whole number that PostgreSQL still refuses
    to read as an integer, and one such cell rolls back the entire migration.
    Deriving the list from db/transform.sql keeps this honest as the casts
    change: purchase order quantities are NUMERIC(9,2) precisely because of it.
    """
    limits = {"INTEGER": 2_147_483_647, "SMALLINT": 32_767}
    problems, casted = [], 0
    for staging, pairs in transform_integer_casts().items():
        name = next(n for n in loader.HEADERS if naming.staging_table(n) == staging)
        positions = {column: naming.staging_columns(name).index(column)
                     for column, _ in pairs}
        dotted = dict.fromkeys(positions, 0)
        largest = dict.fromkeys(positions, 0)
        with loader.required_files()[name].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                for column, index in positions.items():
                    value = fields[index].strip()
                    if not value:
                        continue
                    if "." in value:
                        dotted[column] += 1
                    else:
                        largest[column] = max(largest[column], abs(int(value)))
        for column, sql_type in pairs:
            casted += 1
            if dotted[column]:
                problems.append(f"{staging}.{column}: {dotted[column]} rows carry a "
                                f"decimal point but the transform casts ::{sql_type}")
            if largest[column] > limits[sql_type]:
                problems.append(f"{staging}.{column}: {largest[column]} exceeds "
                                f"the range of {sql_type}")
    assert casted >= 40, f"only {casted} integer casts parsed - the reader broke"
    assert problems == []

-- One-time provisioning of the least-privileged role and database Phase 2
-- needs, with the password collected by psql itself.
--
--     PowerShell (superuser connects; the password is typed at the prompt):
--       & "C:\Program Files\PostgreSQL\18\bin\psql.exe" -v ON_ERROR_STOP=1 `
--           -h 127.0.0.1 -p 5432 -U postgres -f db/provision_interactive.sql
--
-- Why this file sits next to provision.sql rather than replacing it:
-- provision.sql reads ERP_DB_PASSWORD from the environment and puts the value
-- into a CREATE ROLE statement, so the literal crosses the connection and is
-- readable in pg_stat_activity while the statement runs. On a cluster without
-- TLS that literal is plain text on the wire. \password asks psql for the
-- value twice, derives the SCRAM verifier on the client and sends only the
-- verifier, so no secret ever leaves the machine in plaintext.
--
--   * Nothing secret in the repository, the environment, shell history or the
--     server log.
--   * No SUPERUSER, CREATEDB, CREATEROLE, BYPASSRLS or pg_read_server_files.
--     The migration streams the raw files with COPY ... FROM STDIN, so the
--     postmaster never opens data/ and ownership of one database is the whole
--     grant.
--   * No pre-created schema. src/db/migrate.py creates stg, erp and the view
--     inside the transaction it owns; creating them here would put the objects
--     in two places.
--
-- Repeatable. The four conditional \gexec statements create or repair only
-- what is missing, and the password prompt is asked only when the role has no
-- verifier yet, so re-running this on a provisioned server changes nothing and
-- cannot silently break the password in .env. Rotating a password is an
-- explicit act (\password erp), never a side effect of setup.
--
-- \gexec sends the SELECT's result as a statement, which is how a CREATE
-- becomes conditional without a DO block (CREATE DATABASE cannot run inside a
-- transaction block, and a DO block would swallow it). Every query feeding a
-- \gset or an \if returns exactly one row: \gset fails on a zero-row query,
-- and a zero-row condition would quietly read as true.

-- Set here rather than trusted to the command line, because a refusal has to
-- stop the run: without it psql prints the error and executes the rest.
\set ON_ERROR_STOP on

-- CREATE ROLE carries no password clause here, because the value does not
-- exist yet; a role with no verifier still cannot log in under scram-sha-256.
SELECT 'CREATE ROLE erp LOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'erp') \gexec

-- Ownership of the database is what grants CONNECT and CREATE on it, which is
-- everything the migration needs: CREATE SCHEMA, CREATE TABLE, COPY, INSERT.
SELECT 'CREATE DATABASE erp_analytics OWNER erp'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'erp_analytics') \gexec

-- Repair only what a hand-made role or a reassigned owner could have broken.
SELECT 'ALTER ROLE erp LOGIN'
WHERE NOT (SELECT rolcanlogin FROM pg_roles WHERE rolname = 'erp') \gexec

SELECT 'ALTER DATABASE erp_analytics OWNER TO erp'
WHERE (SELECT pg_get_userbyid(datdba) FROM pg_database
       WHERE datname = 'erp_analytics') <> 'erp' \gexec

-- bool_or over no rows is NULL, so the coalesce puts both the missing role and
-- the passwordless role in the "needs a password" branch. pg_authid rather than
-- pg_roles because it is the catalog that exposes rolpassword, and reading it
-- needs the superuser this script already runs as.
SELECT coalesce(bool_or(rolpassword IS NULL), true) AS needs_password
FROM pg_authid WHERE rolname = 'erp' \gset pw_

\if :pw_needs_password
\echo Setting the password for role erp. Type it at the prompt; psql sends
\echo only the SCRAM verifier, never the value you typed.
\password erp
\else
\echo Role erp already has a password, so it is left alone. To rotate it run
\echo psql -c and then the password meta-command, and update the password
\echo value in .env to match.
\endif

-- A role with no verifier is an open analytics database the moment pg_hba has
-- a trust or peer rule, and reporting success in that state is worse than
-- failing here. RAISE rather than \quit, because \quit cannot carry an exit
-- code in psql 18: it always leaves status 0, so an automated run would read a
-- refusal as a provisioning that worked.
SELECT coalesce(bool_or(rolpassword IS NOT NULL), false) AS has_verifier
FROM pg_authid WHERE rolname = 'erp' \gset ok_

\if :ok_has_verifier
\else
\echo ERROR: role erp has no password verifier, so nothing was secured. Set
\echo one before running the migration.
DO $do$ BEGIN RAISE EXCEPTION
  'provisioning refused: role erp has no password verifier'; END $do$;
\endif

-- What the operator should see, and what to check if the migration later
-- reports a privilege error. The verifier itself is never printed: only its
-- type, described by name so that even an md5 value stays off the screen.
SELECT r.rolname, r.rolsuper, r.rolcanlogin, r.rolcreaterole,
       CASE
           WHEN a.rolpassword IS NULL          THEN 'none'
           WHEN a.rolpassword LIKE 'SCRAM-SHA-256$%' THEN 'scram-sha-256'
           WHEN a.rolpassword LIKE 'md5%'      THEN 'md5 - weak, fix password_encryption'
           ELSE 'other'
       END AS verifier_type,
       d.datname, pg_get_userbyid(d.datdba) AS database_owner
FROM pg_authid a
JOIN pg_roles r ON r.rolname = a.rolname
CROSS JOIN pg_database d
WHERE a.rolname = 'erp' AND d.datname = 'erp_analytics';

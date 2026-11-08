-- Staging schema: a faithful, lossless mirror of the seven raw TSV files.
--
-- Every column is TEXT on purpose. COPY never coerces, so a malformed value
-- surfaces later as a readable validation failure instead of an inscrutable
-- import error inside the loader. Column names are the loader.HEADERS names
-- converted mechanically to snake_case (SalesOrderID -> sales_order_id,
-- SubTotal -> sub_total, ModifiedDate -> modified_date) and nothing else is
-- changed: no renaming, no dropping, no type guessing.
--
-- tests/test_db_postgres.py asserts these column lists equal
-- src.data.loader.HEADERS exactly, so this file cannot drift from the file
-- format silently.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.sales_order_header (
    sales_order_id              TEXT,
    revision_number             TEXT,
    order_date                  TEXT,
    due_date                    TEXT,
    ship_date                   TEXT,
    status                      TEXT,
    online_order_flag           TEXT,
    sales_order_number          TEXT,
    purchase_order_number       TEXT,
    account_number              TEXT,
    customer_id                 TEXT,
    sales_person_id             TEXT,
    territory_id                TEXT,
    bill_to_address_id          TEXT,
    ship_to_address_id          TEXT,
    ship_method_id              TEXT,
    credit_card_id              TEXT,
    credit_card_approval_code   TEXT,
    currency_rate_id            TEXT,
    sub_total                   TEXT,
    tax_amt                     TEXT,
    freight                     TEXT,
    total_due                   TEXT,
    comment                     TEXT,
    rowguid                     TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.sales_order_detail (
    sales_order_id              TEXT,
    sales_order_detail_id       TEXT,
    carrier_tracking_number     TEXT,
    order_qty                   TEXT,
    product_id                  TEXT,
    special_offer_id            TEXT,
    unit_price                  TEXT,
    unit_price_discount         TEXT,
    line_total                  TEXT,
    rowguid                     TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.customer (
    customer_id                 TEXT,
    person_id                   TEXT,
    store_id                    TEXT,
    territory_id                TEXT,
    account_number              TEXT,
    rowguid                     TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.product (
    product_id                  TEXT,
    name                        TEXT,
    product_number              TEXT,
    make_flag                   TEXT,
    finished_goods_flag         TEXT,
    color                       TEXT,
    safety_stock_level          TEXT,
    reorder_point               TEXT,
    standard_cost               TEXT,
    list_price                  TEXT,
    size                        TEXT,
    size_unit_measure_code      TEXT,
    weight_unit_measure_code    TEXT,
    weight                      TEXT,
    days_to_manufacture         TEXT,
    product_line                TEXT,
    class                       TEXT,
    style                       TEXT,
    product_subcategory_id      TEXT,
    product_model_id            TEXT,
    sell_start_date             TEXT,
    sell_end_date               TEXT,
    discontinued_date           TEXT,
    rowguid                     TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.product_inventory (
    product_id                  TEXT,
    location_id                 TEXT,
    shelf                       TEXT,
    bin                         TEXT,
    quantity                    TEXT,
    rowguid                     TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.purchase_order_header (
    purchase_order_id           TEXT,
    revision_number             TEXT,
    status                      TEXT,
    employee_id                 TEXT,
    vendor_id                   TEXT,
    ship_method_id              TEXT,
    order_date                  TEXT,
    ship_date                   TEXT,
    sub_total                   TEXT,
    tax_amt                     TEXT,
    freight                     TEXT,
    total_due                   TEXT,
    modified_date               TEXT
);

CREATE TABLE IF NOT EXISTS stg.purchase_order_detail (
    purchase_order_id           TEXT,
    purchase_order_detail_id    TEXT,
    due_date                    TEXT,
    order_qty                   TEXT,
    product_id                  TEXT,
    unit_price                  TEXT,
    line_total                  TEXT,
    received_qty                TEXT,
    rejected_qty                TEXT,
    stocked_qty                 TEXT,
    modified_date               TEXT
);

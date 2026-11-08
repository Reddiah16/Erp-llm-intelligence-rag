-- erp: the normalized business model. One row per real-world thing, so the
-- difference between order-level and line-level money is a property of the
-- schema rather than something every query has to remember.
--
-- Money keeps the precision published in the raw files: sub_total, tax_amt,
-- freight, total_due, unit_price, standard_cost and list_price carry four
-- decimals (measured: 31,366 of 31,465 sales orders have more than two), and
-- sales_order_line.line_total carries six. Nothing is rounded at rest; the
-- display layer rounds. That is what lets the totals reconcile exactly
-- against the source files.
--
-- Integer columns are integer because the text has no decimal point at all
-- (order_qty, quantity, the identifiers), not because the value happened to be
-- whole: purchase_order_line quantities are NUMERIC(9,2) for exactly that
-- reason.

CREATE SCHEMA IF NOT EXISTS erp;

-- ---------------------------------------------------------------- master data

CREATE TABLE IF NOT EXISTS erp.product (
    product_id               INTEGER        PRIMARY KEY,
    name                     VARCHAR(50)    NOT NULL,
    product_number           VARCHAR(25)    NOT NULL UNIQUE,
    make_flag                BOOLEAN        NOT NULL,
    finished_goods_flag      BOOLEAN        NOT NULL,
    color                    VARCHAR(15),
    safety_stock_level       SMALLINT       NOT NULL CHECK (safety_stock_level >= 0),
    reorder_point            SMALLINT       NOT NULL CHECK (reorder_point >= 0),
    standard_cost            NUMERIC(19,4)  NOT NULL DEFAULT 0 CHECK (standard_cost >= 0),
    list_price               NUMERIC(19,4)  NOT NULL DEFAULT 0 CHECK (list_price >= 0),
    size                     VARCHAR(10),
    size_unit_measure_code   VARCHAR(3),
    weight_unit_measure_code VARCHAR(3),
    weight                   NUMERIC(8,2),
    days_to_manufacture      SMALLINT       NOT NULL CHECK (days_to_manufacture >= 0),
    product_line             VARCHAR(2),
    product_class            VARCHAR(2),
    product_style            VARCHAR(2),
    -- No foreign key: ProductSubcategory / ProductModel are not part of the
    -- seven loaded files. An unenforceable FK would be worse than a plain
    -- integer, so the gap is documented instead of faked.
    product_subcategory_id   INTEGER,
    product_model_id         INTEGER,
    sell_start_date          DATE           NOT NULL,
    sell_end_date            DATE,
    rowguid                  UUID           NOT NULL UNIQUE,
    modified_at              TIMESTAMP      NOT NULL
);

CREATE TABLE IF NOT EXISTS erp.customer (
    customer_id     INTEGER      PRIMARY KEY,
    person_id       INTEGER,                    -- NULL for the 701 store-level accounts
    store_id        INTEGER,                    -- NULL for 18,484 individual customers
    territory_id    INTEGER      NOT NULL,       -- SalesTerritory not loaded: no FK
    account_number  VARCHAR(15)  NOT NULL UNIQUE,
    rowguid         UUID         NOT NULL UNIQUE,
    -- Real time-of-day in this dataset (e.g. 2025-09-11 11:15:07.263), so this
    -- column is a TIMESTAMP and not a DATE.
    modified_at     TIMESTAMP    NOT NULL
);

-- --------------------------------------------------------------- sales

CREATE TABLE IF NOT EXISTS erp.sales_order (
    sales_order_id         INTEGER        PRIMARY KEY,
    sales_order_number     VARCHAR(10)    NOT NULL UNIQUE,
    customer_id            INTEGER        NOT NULL REFERENCES erp.customer(customer_id),
    territory_id           INTEGER        NOT NULL,
    sales_person_id        INTEGER,                      -- NULL on the 27,659 online orders
    bill_to_address_id     INTEGER        NOT NULL,
    ship_to_address_id     INTEGER        NOT NULL,
    ship_method_id         SMALLINT       NOT NULL,
    credit_card_id         INTEGER,
    currency_rate_id       INTEGER,
    -- The sales account number, which is NOT customer.account_number: the two
    -- values differ on all 31,465 orders, so both are kept.
    account_number         VARCHAR(15),
    purchase_order_number  VARCHAR(15),
    online_order           BOOLEAN        NOT NULL,
    order_status           SMALLINT       NOT NULL CHECK (order_status BETWEEN 1 AND 6),
    revision_number        SMALLINT       NOT NULL,
    order_date             DATE           NOT NULL,
    due_date               DATE           NOT NULL,
    ship_date              DATE,                          -- NULL means not shipped
    sub_total              NUMERIC(19,4)  NOT NULL,
    tax_amt                NUMERIC(19,4)  NOT NULL,
    freight                NUMERIC(19,4)  NOT NULL,
    total_due              NUMERIC(19,4)  NOT NULL,
    rowguid                UUID           NOT NULL UNIQUE,
    modified_at            TIMESTAMP      NOT NULL,
    -- Verified against the raw files: holds exactly for all 31,465 orders.
    CONSTRAINT sales_order_total_is_composed
        CHECK (total_due = sub_total + tax_amt + freight)
);

CREATE TABLE IF NOT EXISTS erp.sales_order_line (
    sales_order_id           INTEGER        NOT NULL REFERENCES erp.sales_order(sales_order_id),
    sales_order_detail_id    INTEGER        NOT NULL,
    product_id               INTEGER        NOT NULL REFERENCES erp.product(product_id),
    special_offer_id         INTEGER        NOT NULL,       -- SpecialOffer not loaded: no FK
    carrier_tracking_number  VARCHAR(25),
    order_qty                SMALLINT       NOT NULL CHECK (order_qty > 0),
    unit_price               NUMERIC(19,4)  NOT NULL,
    unit_price_discount      NUMERIC(9,4)   NOT NULL DEFAULT 0,
    -- Stored as published, not recomputed: a GENERATED column would re-derive
    -- the amount instead of preserving the source figure.
    line_total               NUMERIC(19,6)  NOT NULL CHECK (line_total >= 0),
    rowguid                  UUID           NOT NULL UNIQUE,
    modified_at              TIMESTAMP      NOT NULL,
    PRIMARY KEY (sales_order_id, sales_order_detail_id)
);

-- --------------------------------------------------------------- inventory

CREATE TABLE IF NOT EXISTS erp.product_inventory (
    product_id    INTEGER   NOT NULL REFERENCES erp.product(product_id),
    location_id   INTEGER   NOT NULL,              -- Inventory.Location not loaded: no FK
    shelf         VARCHAR(5),
    bin           SMALLINT  NOT NULL,
    -- NOT NULL and >= 0 are what make "no record" and "zero on hand" two
    -- different facts: a product is absent from this table, never present with
    -- an empty quantity.
    quantity      INTEGER   NOT NULL CHECK (quantity >= 0),
    rowguid       UUID      NOT NULL UNIQUE,
    modified_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (product_id, location_id)
);

-- -------------------------------------------------------------- purchasing

CREATE TABLE IF NOT EXISTS erp.purchase_order (
    purchase_order_id   INTEGER        PRIMARY KEY,
    revision_number     SMALLINT       NOT NULL,
    -- 1 Pending, 2 Approved, 3 Rejected, 4 Complete (the loader's mapping).
    order_status        SMALLINT       NOT NULL CHECK (order_status BETWEEN 1 AND 4),
    employee_id         SMALLINT       NOT NULL,       -- Employee not loaded: no FK
    vendor_id           INTEGER        NOT NULL,       -- Vendor not loaded: no FK
    ship_method_id      SMALLINT       NOT NULL,
    order_date          DATE           NOT NULL,
    ship_date           DATE,
    sub_total           NUMERIC(19,4)  NOT NULL,
    tax_amt             NUMERIC(19,4)  NOT NULL,
    freight             NUMERIC(19,4)  NOT NULL,
    total_due           NUMERIC(19,4)  NOT NULL,
    modified_at         TIMESTAMP      NOT NULL,       -- no rowguid in this source file
    CONSTRAINT purchase_order_total_is_composed
        CHECK (total_due = sub_total + tax_amt + freight)
);

CREATE TABLE IF NOT EXISTS erp.purchase_order_line (
    purchase_order_id         INTEGER        NOT NULL REFERENCES erp.purchase_order(purchase_order_id),
    purchase_order_detail_id  INTEGER        NOT NULL,
    due_date                  DATE           NOT NULL,
    order_qty                 SMALLINT       NOT NULL CHECK (order_qty > 0),
    product_id                INTEGER        NOT NULL REFERENCES erp.product(product_id),
    unit_price                NUMERIC(19,4)  NOT NULL,
    line_total                NUMERIC(19,4)  NOT NULL CHECK (line_total >= 0),
    -- NUMERIC(9,2) because the feed publishes two decimals ("3.00", ".00"),
    -- which is text an INTEGER cast refuses outright. Every one of the 8,845
    -- source rows is a whole number today (measured: zero rows with a nonzero
    -- fraction), so the values behave like counts while the scale stays as
    -- published. A partial receipt would land here without any change.
    received_qty              NUMERIC(9,2)   NOT NULL CHECK (received_qty >= 0),
    rejected_qty              NUMERIC(9,2)   NOT NULL CHECK (rejected_qty >= 0),
    stocked_qty               NUMERIC(9,2)   NOT NULL CHECK (stocked_qty >= 0),
    modified_at               TIMESTAMP      NOT NULL,
    PRIMARY KEY (purchase_order_id, purchase_order_detail_id),
    CONSTRAINT purchase_order_line_stock_reconciles
        CHECK (stocked_qty = received_qty - rejected_qty)
);

-- ---------------------------------------------------------------- indexes
-- Each one exists for a stated business question and nothing else. The two
-- composite primary keys already start with their header key, so the
-- header -> line joins need no extra index.

CREATE INDEX IF NOT EXISTS sales_order_order_date_idx       ON erp.sales_order (order_date);          -- revenue, monthly trends
CREATE INDEX IF NOT EXISTS sales_order_customer_idx         ON erp.sales_order (customer_id);        -- customers by revenue
CREATE INDEX IF NOT EXISTS sales_order_line_product_idx     ON erp.sales_order_line (product_id);    -- products by revenue
CREATE INDEX IF NOT EXISTS purchase_order_status_idx        ON erp.purchase_order (order_status);    -- pending POs
CREATE INDEX IF NOT EXISTS purchase_order_vendor_idx        ON erp.purchase_order (vendor_id);       -- supplier questions
CREATE INDEX IF NOT EXISTS purchase_order_line_product_idx  ON erp.purchase_order_line (product_id); -- POs by product

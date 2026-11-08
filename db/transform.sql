-- stg -> erp: the only place where raw text becomes typed business data.
--
-- Read top to bottom and the whole migration is visible: every cast is
-- written out, the four dropped source columns are dropped here rather than
-- hidden in Python, and nothing is rounded. NULLIF(x, '') turns the files'
-- empty fields into real NULLs, which is what the raw dataset means by empty
-- (row 1 of Product.csv has 12 empty fields out of 25).
--
-- Dates are cast through TIMESTAMP before DATE on purpose: the source carries
-- a time component ('2022-05-30 00:00:00.000'). It is midnight in every row of
-- every business date column (measured), so truncating loses nothing, and a
-- non-midnight feed would still land on the right calendar day instead of
-- failing the cast.
--
-- ::INTEGER is only used where the source text has no decimal point. The
-- purchase order quantities are published as "3.00", so they are cast to
-- NUMERIC(9,2) and keep that scale; a whole value is not the same thing as
-- integer text.
--
-- There is no BEGIN/COMMIT here on purpose: migrate.py runs this file inside
-- one transaction and commits only after every validation check has passed.
-- Committing here would make a failed check leave a half-loaded database.

TRUNCATE TABLE
    erp.sales_order_line,
    erp.sales_order,
    erp.purchase_order_line,
    erp.purchase_order,
    erp.product_inventory,
    erp.customer,
    erp.product
CASCADE;

-- ------------------------------------------------------------------ product
-- DiscontinuedDate is dropped: it is empty in all 504 source rows.
INSERT INTO erp.product (
    product_id, name, product_number, make_flag, finished_goods_flag, color,
    safety_stock_level, reorder_point, standard_cost, list_price, size,
    size_unit_measure_code, weight_unit_measure_code, weight,
    days_to_manufacture, product_line, product_class, product_style,
    product_subcategory_id, product_model_id, sell_start_date, sell_end_date,
    rowguid, modified_at
)
SELECT
    NULLIF(product_id, '')::INTEGER,
    NULLIF(name, '')::VARCHAR,
    NULLIF(product_number, '')::VARCHAR,
    NULLIF(make_flag, '')::BOOLEAN,
    NULLIF(finished_goods_flag, '')::BOOLEAN,
    NULLIF(color, '')::VARCHAR,
    NULLIF(safety_stock_level, '')::SMALLINT,
    NULLIF(reorder_point, '')::SMALLINT,
    NULLIF(standard_cost, '')::NUMERIC(19,4),
    NULLIF(list_price, '')::NUMERIC(19,4),
    NULLIF(size, '')::VARCHAR,
    NULLIF(size_unit_measure_code, '')::VARCHAR,
    NULLIF(weight_unit_measure_code, '')::VARCHAR,
    NULLIF(weight, '')::NUMERIC(8,2),
    NULLIF(days_to_manufacture, '')::SMALLINT,
    NULLIF(product_line, '')::VARCHAR,
    NULLIF(class, '')::VARCHAR,
    NULLIF(style, '')::VARCHAR,
    NULLIF(product_subcategory_id, '')::INTEGER,
    NULLIF(product_model_id, '')::INTEGER,
    NULLIF(sell_start_date, '')::TIMESTAMP::DATE,
    NULLIF(sell_end_date, '')::TIMESTAMP::DATE,
    NULLIF(rowguid, '')::UUID,
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.product;

-- ----------------------------------------------------------------- customer
INSERT INTO erp.customer (
    customer_id, person_id, store_id, territory_id, account_number,
    rowguid, modified_at
)
SELECT
    NULLIF(customer_id, '')::INTEGER,
    NULLIF(person_id, '')::INTEGER,
    NULLIF(store_id, '')::INTEGER,
    NULLIF(territory_id, '')::INTEGER,
    NULLIF(account_number, '')::VARCHAR,
    NULLIF(rowguid, '')::UUID,
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.customer;

-- --------------------------------------------------------------- sales order
-- Comment and CreditCardApprovalCode are dropped: Comment is empty in all
-- 31,465 rows and the approval code carries no analytical value.
INSERT INTO erp.sales_order (
    sales_order_id, sales_order_number, customer_id, territory_id,
    sales_person_id, bill_to_address_id, ship_to_address_id, ship_method_id,
    credit_card_id, currency_rate_id, account_number, purchase_order_number,
    online_order, order_status, revision_number, order_date, due_date,
    ship_date, sub_total, tax_amt, freight, total_due, rowguid, modified_at
)
SELECT
    NULLIF(sales_order_id, '')::INTEGER,
    NULLIF(sales_order_number, '')::VARCHAR,
    NULLIF(customer_id, '')::INTEGER,
    NULLIF(territory_id, '')::INTEGER,
    NULLIF(sales_person_id, '')::INTEGER,
    NULLIF(bill_to_address_id, '')::INTEGER,
    NULLIF(ship_to_address_id, '')::INTEGER,
    NULLIF(ship_method_id, '')::SMALLINT,
    NULLIF(credit_card_id, '')::INTEGER,
    NULLIF(currency_rate_id, '')::INTEGER,
    NULLIF(account_number, '')::VARCHAR,
    NULLIF(purchase_order_number, '')::VARCHAR,
    NULLIF(online_order_flag, '')::BOOLEAN,
    NULLIF(status, '')::SMALLINT,
    NULLIF(revision_number, '')::SMALLINT,
    NULLIF(order_date, '')::TIMESTAMP::DATE,
    NULLIF(due_date, '')::TIMESTAMP::DATE,
    NULLIF(ship_date, '')::TIMESTAMP::DATE,
    NULLIF(sub_total, '')::NUMERIC(19,4),
    NULLIF(tax_amt, '')::NUMERIC(19,4),
    NULLIF(freight, '')::NUMERIC(19,4),
    NULLIF(total_due, '')::NUMERIC(19,4),
    NULLIF(rowguid, '')::UUID,
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.sales_order_header;

-- ---------------------------------------------------------- sales order line
-- Nothing is dropped: SalesOrderDetailID and SpecialOfferID are exactly the
-- two columns the denormalized clean CSV loses.
INSERT INTO erp.sales_order_line (
    sales_order_id, sales_order_detail_id, product_id, special_offer_id,
    carrier_tracking_number, order_qty, unit_price, unit_price_discount,
    line_total, rowguid, modified_at
)
SELECT
    NULLIF(sales_order_id, '')::INTEGER,
    NULLIF(sales_order_detail_id, '')::INTEGER,
    NULLIF(product_id, '')::INTEGER,
    NULLIF(special_offer_id, '')::INTEGER,
    NULLIF(carrier_tracking_number, '')::VARCHAR,
    NULLIF(order_qty, '')::SMALLINT,
    NULLIF(unit_price, '')::NUMERIC(19,4),
    NULLIF(unit_price_discount, '')::NUMERIC(9,4),
    NULLIF(line_total, '')::NUMERIC(19,6),
    NULLIF(rowguid, '')::UUID,
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.sales_order_detail;

-- ----------------------------------------------------------- inventory
INSERT INTO erp.product_inventory (
    product_id, location_id, shelf, bin, quantity, rowguid, modified_at
)
SELECT
    NULLIF(product_id, '')::INTEGER,
    NULLIF(location_id, '')::INTEGER,
    NULLIF(shelf, '')::VARCHAR,
    NULLIF(bin, '')::SMALLINT,
    NULLIF(quantity, '')::INTEGER,
    NULLIF(rowguid, '')::UUID,
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.product_inventory;

-- ---------------------------------------------------------- purchase order
INSERT INTO erp.purchase_order (
    purchase_order_id, revision_number, order_status, employee_id, vendor_id,
    ship_method_id, order_date, ship_date, sub_total, tax_amt, freight,
    total_due, modified_at
)
SELECT
    NULLIF(purchase_order_id, '')::INTEGER,
    NULLIF(revision_number, '')::SMALLINT,
    NULLIF(status, '')::SMALLINT,
    NULLIF(employee_id, '')::SMALLINT,
    NULLIF(vendor_id, '')::INTEGER,
    NULLIF(ship_method_id, '')::SMALLINT,
    NULLIF(order_date, '')::TIMESTAMP::DATE,
    NULLIF(ship_date, '')::TIMESTAMP::DATE,
    NULLIF(sub_total, '')::NUMERIC(19,4),
    NULLIF(tax_amt, '')::NUMERIC(19,4),
    NULLIF(freight, '')::NUMERIC(19,4),
    NULLIF(total_due, '')::NUMERIC(19,4),
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.purchase_order_header;

-- ------------------------------------------------------ purchase order line
INSERT INTO erp.purchase_order_line (
    purchase_order_id, purchase_order_detail_id, due_date, order_qty,
    product_id, unit_price, line_total, received_qty, rejected_qty,
    stocked_qty, modified_at
)
SELECT
    NULLIF(purchase_order_id, '')::INTEGER,
    NULLIF(purchase_order_detail_id, '')::INTEGER,
    NULLIF(due_date, '')::TIMESTAMP::DATE,
    NULLIF(order_qty, '')::SMALLINT,
    NULLIF(product_id, '')::INTEGER,
    NULLIF(unit_price, '')::NUMERIC(19,4),
    NULLIF(line_total, '')::NUMERIC(19,4),
    NULLIF(received_qty, '')::NUMERIC(9,2),
    NULLIF(rejected_qty, '')::NUMERIC(9,2),
    NULLIF(stocked_qty, '')::NUMERIC(9,2),
    NULLIF(modified_date, '')::TIMESTAMP
FROM stg.purchase_order_detail;

-- erp.stock_position: one row per product, holding the Phase 1 inventory
-- semantics as a query instead of as stored flags.
--
-- Three states are kept apart, and the distinction is the whole point:
--
--   1. Known quantity        a product_inventory row exists, total_quantity is
--                            a real figure and both alert flags are meaningful.
--   2. Confirmed zero        inventory rows exist and their quantities sum to
--                            zero: total_quantity = 0, legitimately flagged.
--   3. Missing record        no inventory row at all: total_quantity is NULL,
--                            has_inventory_record is false, and both flags are
--                            false, because an absent record says nothing about
--                            stock. Such a product is neither healthy nor out.
--
-- COALESCE(total_quantity, 0) appears nowhere, and must never be added: on the
-- canonical dataset it turns 69 below-safety products into 141 by inventing 72
-- stock-outs. The flags spell out the record test instead of relying on NULL
-- comparison quietly dropping rows, so the rule reads as an argument rather
-- than a trick.
--
-- The aggregation happens inside the CTE, before the join: product is the
-- driving table, so a product with two to six inventory rows still produces
-- exactly one row here and the 72 products with no rows are reported instead
-- of vanishing.

CREATE OR REPLACE VIEW erp.stock_position AS
WITH recorded AS (
    SELECT product_id,
           SUM(quantity) AS total_quantity,   -- quantity is NOT NULL, so a real
           COUNT(*)      AS inventory_rows    -- group can never sum to NULL here
    FROM   erp.product_inventory
    GROUP  BY product_id
)
SELECT
    p.product_id,
    p.name,
    p.product_number,
    p.safety_stock_level,
    p.reorder_point,
    r.product_id IS NOT NULL                          AS has_inventory_record,
    r.total_quantity                                  AS total_quantity,
    CASE WHEN r.product_id IS NOT NULL
              AND r.total_quantity < p.safety_stock_level
         THEN TRUE ELSE FALSE END                     AS below_safety,
    CASE WHEN r.product_id IS NOT NULL
              AND r.total_quantity < p.reorder_point
         THEN TRUE ELSE FALSE END                     AS needs_reorder,
    r.inventory_rows                                  AS inventory_row_count
FROM erp.product p
LEFT JOIN recorded r ON r.product_id = p.product_id;

COMMENT ON VIEW erp.stock_position IS
    'On-hand position per product, one row per product. total_quantity is NULL '
    'when the product has no ProductInventory record at all - NULL means unknown, '
    'zero means confirmed empty. Never coalesce it to zero.';

COMMENT ON COLUMN erp.product_inventory.quantity IS
    'On-hand units at one storage location. NOT NULL: an unrecorded product has '
    'no row here rather than a row with a blank quantity.';

COMMENT ON COLUMN erp.sales_order.total_due IS
    'Order-level amount including tax and freight, NUMERIC(19,4) as published. '
    'Sum over erp.sales_order only: joining order lines multiplies this value '
    'by the line count (~23.75x on the canonical dataset).';

COMMENT ON COLUMN erp.sales_order_line.line_total IS
    'Line-level revenue for one order detail row. This is the correct column for '
    'product and per-line revenue; erp.sales_order.total_due is for orders.';

import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

fake = Faker("de_DE")  # German locale for realistic ERP data
random.seed(42)
np.random.seed(42)

SD_DIR = Path("data/sd")
MM_DIR = Path("data/mm")

# SAP-style master data
REGIONS = ["Bayern", "Baden-Württemberg", "NRW", "Hessen", "Hamburg", "Berlin"]
SEGMENTS = ["Automotive", "Manufacturing", "Retail", "Chemical", "Electronics"]
CATEGORIES = ["Raw Material", "Semi-Finished", "Finished Good", "Spare Part", "Packaging"]
UNITS = ["ST", "KG", "L", "M", "PAC"]  # SAP unit codes
ORDER_STATUS = ["Open", "Delivered", "Invoiced", "Cancelled"]
PO_STATUS = ["Open", "Partially Delivered", "Fully Delivered"]


def generate_customer_master(n: int = 100) -> pd.DataFrame:
    log.info(f"Generating {n} customer master records...")
    records = []
    for i in range(1, n + 1):
        records.append({
            "customer_id":   f"CUST-{i:05d}",
            "customer_name": fake.company(),
            "region":        random.choice(REGIONS),
            "segment":       random.choice(SEGMENTS),
            "city":          fake.city(),
            "credit_limit":  round(random.uniform(50000, 500000), 2),
            "currency":      "EUR"
        })
    return pd.DataFrame(records)


def generate_sales_orders(customers: pd.DataFrame, materials: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    log.info(f"Generating {n} sales orders...")
    records = []
    for i in range(1, n + 1):
        customer  = customers.sample(1).iloc[0]
        material  = materials.sample(1).iloc[0]
        qty       = random.randint(1, 200)
        unit_price = round(random.uniform(10, 5000), 2)
        order_date = fake.date_between(start_date="-1y", end_date="today")
        records.append({
            "order_id":       f"SO-{i:06d}",
            "customer_id":    customer["customer_id"],
            "customer_name":  customer["customer_name"],
            "region":         customer["region"],
            "material_id":    material["material_id"],
            "material_desc":  material["description"],
            "quantity":       qty,
            "unit_price":     unit_price,
            "total_value":    round(qty * unit_price, 2),
            "currency":       "EUR",
            "order_date":     order_date.strftime("%Y-%m-%d"),
            "order_month":    order_date.strftime("%Y-%m"),
            "status":         random.choice(ORDER_STATUS)
        })
    return pd.DataFrame(records)


def generate_material_master(n: int = 150) -> pd.DataFrame:
    log.info(f"Generating {n} material master records...")
    records = []
    for i in range(1, n + 1):
        records.append({
            "material_id":   f"MAT-{i:06d}",
            "description":   fake.catch_phrase(),
            "category":      random.choice(CATEGORIES),
            "unit":          random.choice(UNITS),
            "base_price":    round(random.uniform(5, 2000), 2),
            "currency":      "EUR",
            "plant":         f"PLANT-{random.randint(1, 5):02d}",
            "storage_loc":   f"SL-{random.randint(1, 10):02d}"
        })
    return pd.DataFrame(records)


def generate_stock_levels(materials: pd.DataFrame) -> pd.DataFrame:
    log.info("Generating stock levels...")
    records = []
    for _, mat in materials.iterrows():
        safety_stock  = random.randint(10, 100)
        reorder_point = safety_stock + random.randint(10, 50)
        current_stock = random.randint(0, reorder_point * 3)
        records.append({
            "material_id":    mat["material_id"],
            "description":    mat["description"],
            "plant":          mat["plant"],
            "current_stock":  current_stock,
            "safety_stock":   safety_stock,
            "reorder_point":  reorder_point,
            "unit":           mat["unit"],
            "below_safety":   current_stock < safety_stock,
            "needs_reorder":  current_stock < reorder_point
        })
    return pd.DataFrame(records)


def generate_purchase_orders(materials: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    log.info(f"Generating {n} purchase orders...")
    records = []
    for i in range(1, n + 1):
        material   = materials.sample(1).iloc[0]
        qty        = random.randint(50, 1000)
        unit_price = round(random.uniform(5, 1500), 2)
        order_date = fake.date_between(start_date="-6m", end_date="today")
        delivery_date = fake.date_between(start_date="today", end_date="+3m")
        records.append({
            "po_id":          f"PO-{i:06d}",
            "vendor_id":      f"VEND-{random.randint(1, 50):04d}",
            "vendor_name":    fake.company(),
            "material_id":    material["material_id"],
            "material_desc":  material["description"],
            "quantity":       qty,
            "unit_price":     unit_price,
            "total_value":    round(qty * unit_price, 2),
            "currency":       "EUR",
            "order_date":     order_date.strftime("%Y-%m-%d"),
            "delivery_date":  delivery_date.strftime("%Y-%m-%d"),
            "status":         random.choice(PO_STATUS)
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    SD_DIR.mkdir(parents=True, exist_ok=True)
    MM_DIR.mkdir(parents=True, exist_ok=True)

    # Generate MM data first (needed for SD)
    materials = generate_material_master(150)
    stock     = generate_stock_levels(materials)
    po        = generate_purchase_orders(materials, 200)

    # Generate SD data
    customers = generate_customer_master(100)
    orders    = generate_sales_orders(customers, materials, 500)

    # Save all tables
    materials.to_csv(MM_DIR / "material_master.csv", index=False)
    stock.to_csv(MM_DIR    / "stock_levels.csv",     index=False)
    po.to_csv(MM_DIR       / "purchase_orders.csv",  index=False)
    customers.to_csv(SD_DIR / "customer_master.csv", index=False)
    orders.to_csv(SD_DIR   / "sales_orders.csv",     index=False)

    log.info("All ERP tables generated successfully.")
    log.info(f"  SD: {len(customers)} customers | {len(orders)} sales orders")
    log.info(f"  MM: {len(materials)} materials | {len(stock)} stock records | {len(po)} purchase orders")
    log.info(f"  Materials below safety stock: {stock['below_safety'].sum()}")
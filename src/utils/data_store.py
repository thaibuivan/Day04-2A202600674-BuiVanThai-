import hashlib
import json
from datetime import datetime
from pathlib import Path

from core.schemas import OrderLineInput, ProductRecord


class OrderDataStore:
    """
    Student TODO:
    - Load `products.json`.
    - Build lookup helpers for product IDs and normalized search.
    - Save final orders under `artifacts/orders/`.
    """

    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or datetime.now().strftime("%Y-%m-%d")
        
        products_path = self.data_dir / "products.json"
        with open(products_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.products = [ProductRecord(**p) for p in data]
        self.product_index = {p.product_id: p for p in self.products}

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        results = []
        q = query.lower() if query else None
        c = category.lower() if category else None
        rt = [t.lower() for t in required_tags] if required_tags else []

        for p in self.products:
            if in_stock_only and p.stock <= 0:
                continue
            if max_unit_price is not None and p.unit_price > max_unit_price:
                continue
            if c and c != p.category.lower():
                continue
            if rt and not all(t in [tag.lower() for tag in p.tags] for t in rt):
                continue
            if q:
                search_target = f"{p.name} {p.brand} {p.category} {p.description} {' '.join(p.tags)}".lower()
                terms = [t.strip() for t in q.split(",") if t.strip()]
                # Match if ANY term is found in search_target
                if not any(t in search_target for t in terms):
                    continue
            
            results.append({
                "product_id": p.product_id,
                "name": p.name,
                "unit_price": p.unit_price,
                "stock": p.stock,
            })
        return results[:limit]

    def get_product_details(self, product_ids: list[str]) -> list[dict]:
        details = []
        for pid in product_ids:
            if pid in self.product_index:
                p = self.product_index[pid]
                details.append({
                    "product_id": p.product_id,
                    "name": p.name,
                    "unit_price": p.unit_price,
                    "stock": p.stock,
                    "category": p.category,
                    "warranty_months": p.warranty_months,
                    "description": p.description,
                })
        
        # Generate a deterministic token based on sorted IDs
        sorted_ids = sorted(list(set(product_ids)))
        token = hashlib.md5("".join(sorted_ids).encode()).hexdigest()
        
        return {
            "details": details,
            "detail_token": token
        }

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        # Deterministic simulation
        rate = 0.1
        if "vip" in customer_tier.lower() or int(hashlib.md5(seed_hint.encode()).hexdigest(), 16) % 2 == 0:
            rate = 0.2
            
        return {
            "discount_rate": rate,
            "campaign_code": f"PROMO-{int(rate*100)}"
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        product_ids = [item.product_id for item in items]
        expected_token = hashlib.md5("".join(sorted(list(set(product_ids)))).encode()).hexdigest()
        
        if detail_token != expected_token:
            return {"status": "error", "message": "Invalid detail_token. Please call get_product_details first."}
            
        subtotal = 0
        for item in items:
            if item.product_id not in self.product_index:
                return {"status": "error", "message": f"Product ID {item.product_id} not found."}
            
            product = self.product_index[item.product_id]
            if item.quantity > product.stock:
                return {"status": "error", "message": f"Insufficient stock for {product.name} (ID: {item.product_id}). Requested: {item.quantity}, Available: {product.stock}"}
                
            subtotal += product.unit_price * item.quantity
            
        discount_amount = int(subtotal * discount_rate)
        final_total = subtotal - discount_amount
        
        return {
            "status": "success",
            "subtotal": subtotal,
            "discount_amount": discount_amount,
            "final_total": final_total,
        }

    def save_order(
        self,
        *,
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> dict:
        totals = self.calculate_order_totals(
            items=items, 
            detail_token=detail_token, 
            discount_rate=discount_rate
        )
        
        if totals.get("status") == "error":
            return totals
            
        # Deterministic order ID
        seed = f"{customer_email}_{self.today}_{totals['subtotal']}"
        order_id = f"ORD-{hashlib.md5(seed.encode()).hexdigest()[:8].upper()}"
        
        payload = {
            "order_id": order_id,
            "date": self.today,
            "customer": {
                "name": customer_name,
                "phone": customer_phone,
                "email": customer_email,
                "tier": customer_tier,
                "shipping_address": shipping_address,
            },
            "items": [{"product_id": i.product_id, "quantity": i.quantity} for i in items],
            "financials": {
                "subtotal": totals["subtotal"],
                "discount_rate": discount_rate,
                "discount_amount": totals["discount_amount"],
                "final_total": totals["final_total"],
            },
            "campaign_code": campaign_code,
            "notes": notes,
        }
        
        file_path = self.output_dir / f"{order_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            
        return {
            "status": "saved",
            "saved_order": payload,
            "path": str(file_path),
        }

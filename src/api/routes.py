import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from src.model import ModelService

router = APIRouter()

class UserState(BaseModel):
    history: list[int]

_cached_catalog = None

def get_product_catalog():
    global _cached_catalog
    if _cached_catalog is not None:
        return _cached_catalog
    
    dataset_path = "Ecommerce_Dataset.csv"
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            dedup = df.drop_duplicates(subset=['product_id']).head(100).copy()
            products = []
            for idx, row in dedup.iterrows():
                # Store indicator
                pid_str = str(row['product_id']).lower()
                store = "Amazon"
                if "shopee" in pid_str or "218" in pid_str:
                    store = "Shopee"
                elif "walmart" in pid_str or "173" in pid_str:
                    store = "Walmart"
                elif "shein" in pid_str or "404" in pid_str:
                    store = "Shein"
                elif "lazada" in pid_str or "687" in pid_str:
                    store = "Lazada"
                
                products.append({
                    "id": len(products) + 1,
                    "product_id": str(row['product_id']),
                    "title": str(row.get('title', f"Product {len(products)+1}")),
                    "brand": str(row.get('brand', 'Generic')),
                    "price": float(row.get('price', 19.99)),
                    "category": str(row.get('category_code', 'general.goods')),
                    "store": store
                })
            _cached_catalog = products
            return products
        except Exception as e:
            print(f"[WARNING] Catalog load error: {e}")

    # Default 10 products fallback
    fallback = []
    stores = ["Amazon", "Shopee", "Walmart", "Shein", "Lazada"]
    for i in range(1, 11):
        fallback.append({
            "id": i,
            "product_id": f"prod_{i:03d}",
            "title": f"Sample Product #{i}",
            "brand": "Popular Brand",
            "price": round(15.0 + i * 12.5, 2),
            "category": "electronics.gadgets",
            "store": stores[i % len(stores)]
        })
    _cached_catalog = fallback
    return fallback

@router.get("/")
@router.get("/dashboard")
def get_dashboard():
    """Serve the interactive web application & dashboard"""
    dashboard_path = os.path.join("src", "static", "dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard interface file missing")
    return FileResponse(dashboard_path)

@router.get("/api/catalog")
def get_catalog():
    """Return product catalog for interactive selection UI"""
    return JSONResponse(get_product_catalog())

@router.get("/api/metrics")
def get_training_metrics():
    """Return JSON training history & evaluation metrics for the dashboard charts"""
    json_path = os.path.join("model_data", "training_history.json")
    eval_path = os.path.join("model_data", "evaluation_metrics.json")
    
    data = {
        "episodes": 0,
        "rewards": [],
        "epsilon": [],
        "action_counts": [0]*10,
        "summary": {}
    }
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            data["evaluation"] = json.load(f)

    return JSONResponse(data)

@router.post("/recommend")
def get_recommendation(data: UserState):
    """Get product recommendations enriched with full product details"""
    if not ModelService.interpreter:
        try:
            ModelService.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    try:
        recommended_id, q_values = ModelService.predict(data.history)
        catalog = get_product_catalog()
        
        # Build ranked recommendations list
        q_arr = pd.Series(q_values)
        ranked_indices = q_arr.sort_values(ascending=False).index.tolist()
        
        recommendations = []
        for rank, idx in enumerate(ranked_indices[:5], start=1):
            cand_id = idx + 1
            matched = next((p for p in catalog if p["id"] == cand_id), None)
            if not matched:
                matched = {
                    "id": cand_id,
                    "title": f"Recommended Product #{cand_id}",
                    "brand": "Featured Brand",
                    "price": 29.99,
                    "category": "recommended.items",
                    "store": "Amazon"
                }
            
            recommendations.append({
                "rank": rank,
                "product": matched,
                "q_value": float(q_values[idx]),
                "confidence_score": f"{min(99.9, max(50.0, 50.0 + (q_values[idx] / 1000.0))):.1f}%"
            })
            
        return {
            "recommended_id": recommended_id,
            "top_recommendation": recommendations[0]["product"],
            "all_q_values": q_values,
            "ranked_recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
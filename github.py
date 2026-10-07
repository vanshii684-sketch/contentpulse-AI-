"""
Blog Writing Assistant — FastAPI Backend
Provides REST endpoints that power the frontend dashboard.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import os
import sys

# Ensure agents package is importable when running from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import (
    fuse_sources,
    fetch_news,
    fetch_research,
    fetch_social_trends,
    refine_idea,
    generate_blog_outline,
    predict_trends,
    predict_optimal_publishing,
    get_engagement_forecast,
    get_sentiment_timeline,
)

app = FastAPI(
    title="Blog Writing Assistant API",
    description="Agentic AI backend for intelligent blog content management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IdeaRefinementRequest(BaseModel):
    raw_idea: str
    input_type: Optional[str] = "text"   # "text" | "image" | "voice"
    image_description: Optional[str] = None


class PublishingPredictionRequest(BaseModel):
    topic_category: Optional[str] = "AI"


# ---------------------------------------------------------------------------
# Routes — Content Fusion
# ---------------------------------------------------------------------------

@app.get("/api/dashboard", summary="Full dashboard payload (all agents fused)")
async def get_dashboard():
    """
    Returns combined data from all three agents:
    news + research + social + topic suggestions + trends + engagement forecast.
    """
    fusion, engagement, sentiment = await asyncio.gather(
        fuse_sources(),
        get_engagement_forecast(7),
        get_sentiment_timeline(24),
    )
    trend_data = await predict_trends()
    return {
        **fusion,
        "engagement_forecast": engagement,
        "sentiment_timeline": sentiment,
        "category_trends": trend_data,
    }


@app.get("/api/news", summary="Latest curated news articles")
async def get_news(category: str = "all", limit: int = 8):
    return await fetch_news(category=category, limit=limit)


@app.get("/api/research", summary="Research papers & academic content")
async def get_research():
    return await fetch_research()


@app.get("/api/social", summary="Social media trending topics")
async def get_social():
    return await fetch_social_trends()


@app.get("/api/trends", summary="Category trend velocities & 7-day forecasts")
async def get_trends():
    return await predict_trends()


@app.get("/api/engagement-forecast", summary="7-day engagement forecast")
async def get_engagement(days: int = 7):
    return await get_engagement_forecast(days)


@app.get("/api/sentiment-timeline", summary="Hourly sentiment timeline (24h)")
async def get_sentiment(hours: int = 24):
    return await get_sentiment_timeline(hours)


# ---------------------------------------------------------------------------
# Routes — Multimodal Input Agent
# ---------------------------------------------------------------------------

@app.post("/api/refine-idea", summary="Refine a raw blog idea (text/image/voice)")
async def refine_blog_idea(request: IdeaRefinementRequest):
    refined = await refine_idea(
        raw_idea=request.raw_idea,
        input_type=request.input_type,
        image_description=request.image_description,
    )
    return refined


@app.post("/api/generate-outline", summary="Generate a full blog outline from a raw idea")
async def generate_outline(request: IdeaRefinementRequest):
    refined = await refine_idea(
        raw_idea=request.raw_idea,
        input_type=request.input_type,
        image_description=request.image_description,
    )
    outline = await generate_blog_outline(refined)
    return {"refined": refined, "outline": outline}


@app.post("/api/upload-image", summary="Upload an image and extract a blog idea")
async def upload_image(
    file: UploadFile = File(...),
    context: str = Form(default=""),
):
    """
    In production: sends image to a vision model (GPT-4V / Claude 3).
    Here: returns a mock image analysis with blog suggestions.
    """
    filename = file.filename or "uploaded_image"
    mock_description = (
        f"The uploaded image '{filename}' appears to show technology-related content. "
        "Visual elements suggest themes of innovation, digital transformation, and human-AI collaboration."
    )
    combined_idea = context if context else "Blog idea inspired by uploaded image"
    refined = await refine_idea(
        raw_idea=combined_idea,
        input_type="image",
        image_description=mock_description,
    )
    return {
        "image_filename": filename,
        "detected_themes": ["technology", "innovation", "digital transformation"],
        "refined_idea": refined,
    }


# ---------------------------------------------------------------------------
# Routes — Predictive Agent
# ---------------------------------------------------------------------------

@app.post("/api/predict-publishing", summary="Optimal publish day, time & hashtags")
async def predict_publishing(request: PublishingPredictionRequest):
    return await predict_optimal_publishing(topic_category=request.topic_category)


@app.get("/api/topic-suggestions", summary="AI-ranked blog topic suggestions")
async def get_topic_suggestions():
    data = await fuse_sources()
    return data["topic_suggestions"]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "blog-assistant-api"}

# ---------------------------------------------------------------------------
# Serve frontend (must be last)
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

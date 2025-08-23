# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
from typing import Optional
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

from utils.solar_client import SolarClient
from models.summarizer import ContentSummarizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RESOLAR AI",
    description="당신의 완벽한 자료 조사를 위하여",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

solar_client = SolarClient(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

summarizer = ContentSummarizer(solar_client)

class AnalyzeRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200, description="자료 조사 폴더명")
    title: str = Field(..., min_length=1, max_length=500, description="웹페이지 제목")
    url: str = Field(..., description="웹페이지 URL")
    content: str = Field(..., min_length=10, description="웹페이지 내용")
    timestamp: str = Field(..., description="수집 시간 (ISO 8601 형식)")
    
    @validator('url')
    def validate_url(cls, v):
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('유효하지 않은 URL 형식입니다')
        return v

class AnalyzeResponse(BaseModel):
    success: bool
    summary: str = Field(..., description="3-5줄 요약")
    importance: float = Field(..., ge=1.0, le=10.0, description="중요도 점수 (1-10)")

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze_content(request: AnalyzeRequest):
    """
    웹페이지 컨텐츠 분석 및 요약
    """
    try:
        
        result = summarizer.analyze_content(
            subject=request.subject,
            title=request.title,
            url=request.url,
            content=request.content,
            timestamp=request.timestamp
        )
        
        if not result or not result.get("success"):
            logger.error("컨텐츠 분석 실패")
            raise HTTPException(
                status_code=500,
                detail="컨텐츠 분석 중 오류가 발생했습니다"
            )
        
        logger.info(f"분석 완료: 중요도 {result['importance']}")
        return AnalyzeResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 요청 처리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
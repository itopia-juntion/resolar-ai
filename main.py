# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
import uuid

load_dotenv()

from utils.solar_client import SolarClient
from models.summarizer import ContentSummarizer
from models.rag_search import RAGSearch
from config import get_settings
from report_agent import app as report_agent_app  

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

settings = get_settings()

solar_client = SolarClient(
    api_key=settings.upstage_api_key,
    base_url="https://api.upstage.ai/v1"
)

# Qdrant 클라이언트 초기화
qdrant_client = QdrantClient(
    url=settings.qdrant_host,
    api_key=settings.qdrant_api_key
)
# 요약 및 RAG 클래스 초기화
summarizer = ContentSummarizer(solar_client)
# RAGSearch는 이제 생성자에서 collection_name을 받지 않습니다.
rag_search = RAGSearch(solar_client, qdrant_client)

class AnalyzeRequest(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    subject: str = Field(..., min_length=1, max_length=200, description="자료 조사 폴더명")
    title: str = Field(..., min_length=1, max_length=500, description="웹페이지 제목")
    url: str = Field(..., description="웹페이지 URL")
    content: str = Field(..., min_length=10, description="웹페이지 내용")
    timestamp: str = Field(..., description="수집 시간 (ISO 8601 형식)")
    id: int = Field(..., ge=1, description="웹페이지 고유 ID")
    
    @validator('url')
    def validate_url(cls, v):
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('유효하지 않은 URL 형식입니다')
        return v

class AnalyzeResponse(BaseModel):
    success: bool
    summary: str = Field(..., description="3-5줄 요약")
    importance: float = Field(..., ge=1.0, le=10.0, description="중요도 점수 (1-10)")

class SearchRequest(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    query: str = Field(..., min_length=1, description="검색 쿼리")
    subject: Optional[str] = Field(None, description="폴더명 (선택 사항)")
    limit: int = Field(5, ge=1, le=20, description="검색 결과 개수")
    
class SearchResponse(BaseModel):
    success: bool
    answer: str = Field(..., description="RAG 기반 답변")
    url: str = Field(..., description="가장 관련성 높은 문서의 URL")
    title: str = Field(..., description="가장 관련성 높은 문서의 제목")
    id: int = Field(..., description="가장 관련성 높은 문서의 고유 ID")

class GenerateReportRequest(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    subject: str = Field(..., min_length=1, description="보고서 생성 주제")

class GenerateReportResponse(BaseModel):
    success: bool
    title: str
    introduction: str
    body: str
    conclusion: str

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
            timestamp=request.timestamp,
            id=request.id
        )
        
        if not result or not result.get("success"):
            logger.error("컨텐츠 분석 실패")
            raise HTTPException(
                status_code=500,
                detail="컨텐츠 분석 중 오류가 발생했습니다"
            )

        # 문서 벡터 저장
        rag_search.save_document(
            user_id=request.user_id,
            doc_id=request.id, 
            subject=request.subject,
            title=request.title,
            url=request.url,
            summary=result["summary"]
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
  
@app.post("/api/v1/papersearch", response_model=AnalyzeResponse)
def analyze_paper(request: AnalyzeRequest):
    """
    대규모 요약
    """
    try:
        result = summarizer.analyze_paper(
            subject=request.subject,
            title=request.title,
            url=request.url,
            content=request.content,
            timestamp=request.timestamp,
            id=request.id
        )
        
        if not result or not result.get("success"):
            logger.error("컨텐츠 분석 실패")
            raise HTTPException(
                status_code=500,
                detail="컨텐츠 분석 중 오류가 발생했습니다"
            )

        # 문서 벡터 저장
        rag_search.save_document(
            user_id=request.user_id,
            doc_id=request.id, 
            subject=request.subject,
            title=request.title,
            url=request.url,
            summary=result["summary"]
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
        
              
@app.post("/api/v1/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    """
    RAG 기반 문서 검색 및 답변 생성
    """
    try:
        search_results = rag_search.search_documents(
            user_id=request.user_id,
            query=request.query,
            subject=request.subject,
            limit=request.limit
        )
        
        rag_answer = rag_search.generate_rag_answer(
            query=request.query,
            search_results=search_results
        )
        
        if not rag_answer.get("success"):
            raise HTTPException(
                status_code=500,
                detail="RAG 답변 생성 중 오류가 발생했습니다."
            )
            
        return SearchResponse(**rag_answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"검색 요청 처리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다."
        )

@app.post("/api/v1/generate-report", response_model=GenerateReportResponse)
async def generate_report_from_subject(request: GenerateReportRequest):
    """
    LangGraph 에이전트를 사용하여 최종 보고서 생성
    """
    try:
        logger.info(f"보고서 생성 요청 시작: {request.subject}")
        # LangGraph 에이전트 워크플로 실행
        final_state = await report_agent_app.ainvoke({"user_id": request.user_id, "subject": request.subject})
        
        report = final_state.get("report")
        
        if not report or not isinstance(report, dict):
             raise HTTPException(
                status_code=500,
                detail="보고서 생성 에이전트가 올바른 결과를 반환하지 못했습니다."
            )

        return GenerateReportResponse(
            success=True,
            title=report.get("title", "제목 없음"),
            introduction=report.get("introduction", ""),
            body=report.get("body", ""),
            conclusion=report.get("conclusion", "")
        )
    except Exception as e:
        logger.error(f"보고서 생성 요청 처리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="보고서 생성 중 서버 내부 오류가 발생했습니다"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
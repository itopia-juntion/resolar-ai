# report_agent.py

import os
import logging
from typing import Dict, TypedDict, List, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 프로젝트 내부 모듈 임포트
from utils.solar_client import SolarClient
from models.rag_search import RAGSearch
from utils.web_search_tool import WebSearchTool
from models.report_generator import ReportGenerator
from config import get_settings

load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# 상태 정의
class GraphState(TypedDict):
    """
    LangGraph 상태를 정의하는 클래스
    """
    user_id: str  # user_id 필드 추가
    subject: str
    db_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    report: Dict[str, Any]

# 도구 및 클라이언트 초기화
settings = get_settings()
solar_client = SolarClient(
    api_key=settings.upstage_api_key,
    base_url="https://api.upstage.ai/v1"
)
qdrant_client = QdrantClient(
    url=settings.qdrant_host,
    api_key=settings.qdrant_api_key
)

# RAGSearch는 이제 생성자에서 collection_name을 받지 않습니다.
rag_search_tool = RAGSearch(solar_client, qdrant_client)
web_search_tool = WebSearchTool(brave_search_api_key=settings.brave_search_api_key) 
report_generator = ReportGenerator(solar_client)

### 노드(Node) 정의

def get_db_content(state: GraphState):
    """벡터 DB에서 문서들을 검색하고, 다음 단계(노드 이름)를 결정하는 노드"""
    user_id = state["user_id"] # user_id 받기
    subject = state["subject"]
    db_results = rag_search_tool.get_all_documents_by_subject(user_id, subject) # user_id 전달
    
    # 여기서 다음 단계를 결정하는 로직을 수행
    if not db_results or len(db_results) < 5:
        logger.info("DB 자료가 충분하지 않아 웹 검색을 진행합니다.")
        return {"db_results": db_results, "next": "web_search"}
    else:
        logger.info("DB 자료가 충분하여 웹 검색을 건너뜁니다.")
        return {"db_results": db_results, "next": "generate_report"}

def search_web(state: GraphState):
    """웹을 검색하는 노드"""
    subject = state["subject"]
    web_results = web_search_tool.search(query=subject + " 최신 동향")
    return {"web_results": web_results}

def generate_report(state: GraphState):
    """최종 보고서를 생성하는 노드"""
    subject = state["subject"]
    db_results = state["db_results"]
    web_results = state["web_results"]
    
    report_output = report_generator.generate_final_report(subject, db_results, web_results)
    
    if report_output["success"]:
        return {"report": report_output["report"]}
    else:
        # 실패 시 에러 처리
        return {"report": {"title": "보고서 생성 실패", "body": report_output["error"]}}

### 그래프 구축

workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("get_db_content", get_db_content)
workflow.add_node("search_web", search_web)
workflow.add_node("generate_report", generate_report)

# 시작점 설정
workflow.set_entry_point("get_db_content")

# 조건부 엣지 (의사결정)
workflow.add_conditional_edges(
    "get_db_content",
    lambda state: state["next"],
    {
        "web_search": "search_web",
        "generate_report": "generate_report"
    }
)

# 일반 엣지 연결
workflow.add_edge("search_web", "generate_report")
workflow.add_edge("generate_report", END)

# 그래프 컴파일
app = workflow.compile()
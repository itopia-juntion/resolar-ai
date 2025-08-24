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
    """벡터 DB에서 문서들을 검색하는 노드"""
    user_id = state["user_id"]
    subject = state["subject"]
    db_results = rag_search_tool.get_all_documents_by_subject(user_id, subject)
    
    logger.info(f"DB에서 {len(db_results)}개 문서 검색 완료")
    
    # 상태 업데이트 시 기존 상태를 보존하고 db_results만 업데이트
    return {"db_results": db_results}

def decide_next_step(state: GraphState):
    """다음 단계를 결정하는 함수 (조건부 엣지용)"""
    db_results = state["db_results"]
    
    if not db_results or len(db_results) < 5:
        logger.info("DB 자료가 충분하지 않아 웹 검색을 진행합니다.")
        return "search_web"
    else:
        logger.info("DB 자료가 충분하여 웹 검색을 건너뜁니다.")
        return "generate_report"

def search_web(state: GraphState):
    """웹을 검색하는 노드"""
    subject = state["subject"]
    web_results = web_search_tool.search(query=subject + " 최신 동향")
    logger.info(f"웹에서 {len(web_results)}개 결과 검색 완료")
    return {"web_results": web_results}

def generate_report(state: GraphState):
    """최종 보고서를 생성하는 노드"""
    subject = state["subject"]
    db_results = state["db_results"]
    web_results = state.get("web_results", [])  # web_results가 없을 경우 빈 리스트로 처리
    
    logger.info(f"보고서 생성 시작: DB 문서 {len(db_results)}개, 웹 결과 {len(web_results)}개")
    
    report_output = report_generator.generate_final_report(subject, db_results, web_results)
    
    if report_output["success"]:
        logger.info("보고서 생성 성공")
        return {"report": report_output["report"]}
    else:
        logger.error(f"보고서 생성 실패: {report_output.get('error', '알 수 없는 오류')}")
        return {"report": {
            "title": "보고서 생성 실패", 
            "introduction": "",
            "body": f"오류 발생: {report_output.get('error', '알 수 없는 오류')}",
            "conclusion": ""
        }}

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
    decide_next_step,  # 별도 함수로 분리
    {
        "search_web": "search_web",
        "generate_report": "generate_report"
    }
)

# 일반 엣지 연결
workflow.add_edge("search_web", "generate_report")
workflow.add_edge("generate_report", END)

# 그래프 컴파일
app = workflow.compile()
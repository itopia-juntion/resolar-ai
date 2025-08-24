# models/rag_search.py
import json
import logging
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from utils.solar_client import SolarClient

logger = logging.getLogger(__name__)

class RAGSearch:
    def __init__(
        self, 
        solar_client: SolarClient, 
        qdrant_client: QdrantClient
    ):
        self.solar_client = solar_client
        self.qdrant_client = qdrant_client

    def _get_collection_name(self, user_id: str) -> str:
        """사용자 ID를 기반으로 컬렉션 이름을 생성합니다."""
        return f"user-{user_id}"

    def _ensure_collection_exists(self, collection_name: str):
        """컬렉션이 존재하지 않으면 새로 생성합니다."""
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Qdrant 컬렉션 '{collection_name}'이 이미 존재합니다.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
            )
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="subject",
                field_schema="keyword"
            )
            logger.info(f"Qdrant 컬렉션 '{collection_name}'을 새로 생성했습니다.")

    def save_document(
        self,
        user_id: str,
        doc_id: int,
        subject: str,
        title: str,
        url: str,
        summary: str,
    ) -> bool:
        """
        문서를 임베딩하여 Qdrant에 저장
        """
        try:
            collection_name = self._get_collection_name(user_id)
            self._ensure_collection_exists(collection_name)
            
            embedding = self.solar_client.generate_embedding(texts=[summary])[0]
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "subject": subject,
                    "title": title,
                    "url": url,
                    "summary": summary
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.info(f"문서 '{title}' 저장 성공")
            return True
        except Exception as e:
            logger.error(f"문서 저장 오류: {str(e)}")
            return False

    def search_documents(
        self, 
        user_id: str,
        query: str, 
        subject: Optional[str] = None, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        쿼리를 기반으로 관련 문서 검색
        """
        try:
            collection_name = self._get_collection_name(user_id)
            # 검색 전에 컬렉션 존재 여부 확인
            self._ensure_collection_exists(collection_name) 
            
            query_vector = self.solar_client.generate_embedding(texts=[query])[0]
            
            search_filter = None
            if subject:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="subject",
                            match=MatchValue(value=subject)
                        )
                    ]
                )
            
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "url": hit.payload["url"],
                    "title": hit.payload["title"],
                    "relevance": hit.score,
                    "snippet": hit.payload["summary"],
                })
                
            logger.info(f"검색 완료: {len(results)}개 결과 반환")
            return results
        except Exception as e:
            logger.error(f"검색 오류: {str(e)}")
            return []

    def generate_rag_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        검색 결과와 쿼리를 바탕으로 답변 생성
        """
        # 이 함수는 RAG 답변만 생성하므로 user_id가 필요하지 않습니다.
        if not search_results:
            return {
                "success": True,
                "answer": "죄송합니다. 관련 자료를 찾을 수 없습니다.",
                "url": "",
                "title": "",
                "id": None
            }

        # 가장 관련성 높은 첫 번째 문서만 사용
        first_doc = search_results[0]
        
        context = f"제목: {first_doc['title']}\n내용: {first_doc['snippet']}\n\n"
        
        prompt = f"""
            다음은 검색된 자료입니다. 이 자료를 참고하여 사용자의 질문에 답변해주세요.

            [참고 자료]
            {context}

            [사용자 질문]
            {query}

            요구사항:
            1. 참고 자료를 바탕으로 질문에 대한 답변을 생성해주세요.
            2. 답변과 신뢰도를 JSON 형식으로 반환해주세요.
            """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "생성된 답변"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "신뢰도 (0.00-1.00)"
                        }
                    },
                    "required": ["answer", "confidence"]
                }
            }
        }
        
        try:
            response = self.solar_client.client.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "success": True,
                "answer": result["answer"],
                "url": first_doc["url"],
                "title": first_doc["title"],
                "id": first_doc["id"]
            }
        except Exception as e:
            logger.error(f"RAG 답변 생성 오류: {str(e)}")
            return {
                "success": False,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "url": "",
                "title": "",
                "id": None
            }

    def get_all_documents_by_subject(self, user_id: str, subject: str) -> List[Dict[str, Any]]:
        """
        특정 주제에 해당하는 모든 문서를 Qdrant에서 가져옵니다.
        """
        try:
            collection_name = self._get_collection_name(user_id)
            self._ensure_collection_exists(collection_name)
            
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="subject",
                        match=MatchValue(value=subject)
                    )
                ]
            )
            
            records, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=100
            )
            
            documents = []
            for record in records:
                documents.append({
                    "id": record.id,
                    "url": record.payload["url"],
                    "title": record.payload["title"],
                    "summary": record.payload["summary"]
                })
            
            logger.info(f"주제 '{subject}'에 대해 {len(documents)}개 문서 검색 완료.")
            return documents
        except Exception as e:
            logger.error(f"전체 문서 검색 오류: {str(e)}")
            return []
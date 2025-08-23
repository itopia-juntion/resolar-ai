# models/rag_search.py

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
        qdrant_client: QdrantClient, 
        collection_name: str
    ):
        self.solar_client = solar_client
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
            logger.info(f"Qdrant 컬렉션 '{self.collection_name}'이 이미 존재합니다.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Qdrant 컬렉션 '{self.collection_name}'을 새로 생성했습니다.")

    def save_document(
        self,
        doc_id: str,
        subject: str,
        title: str,
        url: str,
        content: str
    ) -> bool:
        """
        문서를 임베딩하여 Qdrant에 저장
        """
        try:
            embedding = self.solar_client.generate_embedding(texts=[content])[0]
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "subject": subject,
                    "title": title,
                    "url": url,
                    "content": content
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"문서 '{title}' 저장 성공")
            return True
        except Exception as e:
            logger.error(f"문서 저장 오류: {str(e)}")
            return False

    def search_documents(self, query: str, subject: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리를 기반으로 관련 문서 검색
        """
        try:
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
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "url": hit.payload["url"],
                    "title": hit.payload["title"],
                    "relevance": hit.score,
                    "snippet": hit.payload["content"][:200] + "..."
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
        if not search_results:
            return {
                "success": True,
                "answer": "죄송합니다. 관련 자료를 찾을 수 없습니다.",
                "related_urls": [],
                "confidence": 0.0
            }

        context = ""
        urls = []
        for doc in search_results:
            context += f"제목: {doc['title']}\n내용: {doc['snippet']}\n\n"
            urls.append({
                "url": doc["url"],
                "title": doc["title"],
                "relevance": doc["relevance"],
                "snippet": doc["snippet"]
            })
        
        prompt = f"""
            다음은 검색된 자료들입니다. 이 자료들을 참고하여 사용자의 질문에 답변해주세요.

            [참고 자료]
            {context}

            [사용자 질문]
            {query}

            요구사항:
            1. 참고 자료를 바탕으로 질문에 대한 답변을 생성해주세요.
            2. 답변에 대한 신뢰도를 0.0 ~ 1.0 사이의 소수점 2자리로 평가해주세요.
            3. 관련 자료를 찾을 수 없거나 답변이 불확실할 경우, 솔직하게 답변할 수 없다고 말해주세요.
            4. 답변과 신뢰도를 JSON 형식으로 반환해주세요.
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
                "related_urls": urls,
                "confidence": round(result["confidence"], 2)
            }
        except Exception as e:
            logger.error(f"RAG 답변 생성 오류: {str(e)}")
            return {
                "success": False,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "related_urls": [],
                "confidence": 0.0
            }
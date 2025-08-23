# utils/solar_client.py

import json
import logging
from typing import Dict, Any, List
from openai import OpenAI  

logger = logging.getLogger(__name__)

class SolarClient:
    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def generate_summary_and_importance(
        self, 
        subject: str, 
        title: str, 
        content: str
    ) -> Dict[str, Any]:
        """
        Structured Response로 요약 및 중요도 생성
        """
        prompt = f"""
다음 웹페이지 내용을 분석해주세요:

제목: {title}
폴더: {subject}
내용: {content}

요구사항:
1. 핵심 내용을 3줄로 명확히 요약 (각 줄은 마침표로 끝남)
2. 정보 가치, 신뢰성을 고려한 중요도 점수 (1-10점, 소수점 1자리)
        """
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "content_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "3줄 요약 내용"
                        },
                        "importance": {
                            "type": "number",
                            "description": "중요도 점수 (1.0-10.0)"
                        }
                    },
                    "required": ["summary", "importance"]
                }
            }
        }
        
        try:
            logger.info("Solar Pro 2 API 호출")
            
            response = self.client.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("Solar Pro 2 API 성공")
            
            return result
            
        except Exception as e:
            logger.error(f"Solar Pro 2 API 오류: {str(e)}")
            return {
                "summary": "요약 생성 중 오류가 발생했습니다.",
                "importance": 5.0
            }

    def generate_paper_summary(
        self, 
        subject: str, 
        title: str, 
        content: str
    ) -> Dict[str, Any]:
        """
        Structured Response로 요약 및 중요도 생성
        """
        prompt = f"""
        다음 논문의 중요한 사항을 사용자가 하이라이팅 친 내용입니다.

        제목: {title}
        폴더: {subject}
        내용: {content}

        요구사항:
        1. 최대한 원본 내용을 보존하되, 중복되는 내용을 줄여주세요.
        2. 환각 없이 작성해주세요.
        """
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "content_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "요약 내용"
                        },
                        "importance": {
                            "type": "number",
                            "description": "중요도 점수 (1.0-10.0)"
                        }
                    },
                    "required": ["summary", "importance"]
                }
            }
        }
        
        try:
            logger.info("Solar Pro 2 API 호출")
            
            response = self.client.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("Solar Pro 2 API 성공")
            
            return result
            
        except Exception as e:
            logger.error(f"Solar Pro 2 API 오류: {str(e)}")
            return {
                "summary": "요약 생성 중 오류가 발생했습니다.",
                "importance": 5.0
            }


    def generate_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 임베딩 벡터로 변환
        """
        try:
            response = self.client.embeddings.create(
                model="solar-embedding-1-large-passage",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Solar Embedding API 오류: {str(e)}")
            return []
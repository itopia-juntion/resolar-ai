# models/summarizer.py

import logging
import re
from typing import Dict, Any, Optional
from utils.solar_client import SolarClient

logger = logging.getLogger(__name__)

class ContentSummarizer:
    def __init__(self, solar_client: SolarClient):
        self.solar_client = solar_client
    
    def preprocess_content(self, content: str) -> str:
        """
        텍스트 전처리
        """
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        if len(content) > 8000:
            content = content[:8000]
        
        return content

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        기본적인 결과 검증
        """
        if not isinstance(result, dict):
            return False
        
        if "summary" not in result or "importance" not in result:
            return False
        
        if not isinstance(result["summary"], str) or not result["summary"].strip():
            return False
        
        try:
            importance = float(result["importance"])
            if not (1.0 <= importance <= 10.0):
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def analyze_content(
        self, 
        subject: str, 
        title: str, 
        url: str, 
        content: str, 
        timestamp: str,
        id: int
    ) -> Optional[Dict[str, Any]]:
        """
        컨텐츠 분석 및 요약
        """
        try:
            logger.info(f"컨텐츠 분석 시작: {title}")
            
            processed_content = self.preprocess_content(content)
            
            result = self.solar_client.generate_summary_and_importance(
                subject, title, processed_content
            )
            
            if not self.validate_result(result):
                logger.error("결과 검증 실패")
                return {
                    "success": False,
                    "summary": "결과 검증에 실패했습니다.",
                    "importance": 5.0
                }
            
            final_result = {
                "success": True,
                "summary": result["summary"],
                "importance": round(result["importance"], 1)
            }
            
            logger.info(f"컨텐츠 분석 완료: 중요도 {final_result['importance']}")
            return final_result
            
        except Exception as e:
            logger.error(f"컨텐츠 분석 오류: {str(e)}")
            return {
                "success": False,
                "summary": "분석 중 오류가 발생했습니다.",
                "importance": 5.0
            }
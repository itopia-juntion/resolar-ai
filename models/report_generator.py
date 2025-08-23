# models/report_generator.py

import json
import logging
from typing import Dict, Any, List
from utils.solar_client import SolarClient

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, solar_client: SolarClient):
        self.solar_client = solar_client
    
    def generate_final_report(
        self, 
        subject: str, 
        db_content: List[Dict[str, Any]], 
        web_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        벡터 DB와 웹 검색 결과를 결합하여 보고서를 생성합니다.
        """
        
        db_text = "\n\n".join([f"제목: {doc['title']}\n내용: {doc['summary']}" for doc in db_content])
        web_text = "\n\n".join([f"제목: {doc['title']}\n내용: {doc['snippet']}" for doc in web_content])

        prompt = f"""
다음은 '{subject}' 주제에 대한 자료들입니다. 이 자료들을 종합하여 최종 보고서를 작성해주세요.

[내부 지식(벡터 DB)]
{db_text}

[외부 지식(웹 검색)]
{web_text}

보고서는 다음 요구사항을 충족해야 합니다:
1. 보고서 제목을 생성하세요.
2. 보고서 본문은 서론, 본론(세부 내용), 결론으로 구성하세요.
3. 모든 내용은 제공된 자료를 바탕으로 작성해야 합니다.
4. 모든 내용은 한국어로 작성하세요.

JSON 형식으로 반환하세요:
{{
  "title": "보고서 제목",
  "introduction": "서론 내용",
  "body": "본론 내용",
  "conclusion": "결론 내용"
}}
"""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "report_generation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "introduction": {"type": "string"},
                        "body": {"type": "string"},
                        "conclusion": {"type": "string"}
                    },
                    "required": ["title", "introduction", "body", "conclusion"]
                }
            }
        }
        
        try:
            logger.info("보고서 생성 시작")
            response = self.solar_client.client.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=0.5,
                max_tokens=2000
            )
            result = json.loads(response.choices[0].message.content)
            logger.info("보고서 생성 완료")
            return {"success": True, "report": result}
        except Exception as e:
            logger.error(f"보고서 생성 오류: {str(e)}")
            return {"success": False, "error": str(e)}
import logging
from typing import List, Dict, Any
from langchain_community.tools import BraveSearch
from config import get_settings

logger = logging.getLogger(__name__)

class WebSearchTool:
    def __init__(self):
        self.search_tool = get_settings.brave_search_api_key

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        주어진 쿼리로 웹을 검색하고 결과를 반환합니다.
        """
        logger.info(f"웹 검색 시작: '{query}'")
        try:
            # BraveSearch는 JSON 결과를 반환합니다.
            raw_results = self.search_tool.run(query)
            
            results = []
            # 원하는 형식으로 결과 파싱
            for res in raw_results['web']['results']:
                results.append({
                    "title": res['title'],
                    "url": res['url'],
                    "snippet": res['description']
                })
            
            logger.info(f"웹 검색 완료. {len(results)}개 결과 반환")
            return results[:num_results]
        except Exception as e:
            logger.error(f"웹 검색 오류: {str(e)}")
            return []

# utils/web_search_tool.py

import json
import logging
from typing import List, Dict, Any
from langchain_community.tools import BraveSearch

logger = logging.getLogger(__name__)

# BraveSearch API 키를 환경 변수에서 가져옵니다.
from dotenv import load_dotenv
load_dotenv()

class WebSearchTool:
    def __init__(self, brave_search_api_key: str):
        self.search_tool = BraveSearch(api_key=brave_search_api_key)

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        주어진 쿼리로 웹을 검색하고 결과를 반환합니다.
        """
        logger.info(f"웹 검색 시작: '{query}'")
        try:
            raw_results_str = self.search_tool.run(query)
            logger.debug(f"Raw search results: {raw_results_str}")
            
            # JSON 파싱
            raw_results = json.loads(raw_results_str)
            
            results = []
            
            # raw_results의 타입에 따라 처리 방식 결정
            if isinstance(raw_results, list):
                # raw_results가 리스트인 경우
                if raw_results:
                    # 첫 번째 요소가 딕셔너리이고 'web' 키를 가진 경우
                    first_item = raw_results[0]
                    if isinstance(first_item, dict) and 'web' in first_item:
                        web_data = first_item['web']
                    else:
                        # 직접 결과 리스트인 경우
                        web_data = raw_results
                else:
                    web_data = []
            elif isinstance(raw_results, dict):
                # raw_results가 딕셔너리인 경우
                web_data = raw_results.get('web', [])
            else:
                logger.warning(f"예상치 못한 raw_results 타입: {type(raw_results)}")
                web_data = []
            
            # web_data에서 실제 검색 결과 추출
            web_results_list = []
            if isinstance(web_data, list):
                if web_data and isinstance(web_data[0], dict) and 'results' in web_data[0]:
                    web_results_list = web_data[0]['results']
                else:
                    # web_data 자체가 결과 리스트인 경우
                    web_results_list = web_data
            elif isinstance(web_data, dict) and 'results' in web_data:
                web_results_list = web_data['results']
            
            # 결과 포맷팅
            for res in web_results_list:
                if isinstance(res, dict):
                    results.append({
                        "title": res.get('title', '제목 없음'),
                        "url": res.get('url', ''),
                        "snippet": res.get('description', res.get('snippet', '설명 없음'))
                    })
            
            logger.info(f"웹 검색 완료. {len(results)}개 결과 반환")
            return results[:num_results]
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {str(e)}")
            logger.error(f"Raw response: {raw_results_str}")
            return []
        except Exception as e:
            logger.error(f"웹 검색 오류: {str(e)}")
            logger.error(f"오류 타입: {type(e).__name__}")
            return []
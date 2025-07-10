# agent_core/agents/tools/retrievers/patent_retriever.py
# 专利检索器实现

import asyncio
import concurrent.futures
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

@dataclass
class Patent:
    """专利数据结构"""
    patent_id: str
    title: str
    abstract: str
    inventors: List[str]
    assignee: str
    filing_date: str
    publication_date: str
    patent_type: str
    status: str
    classifications: List[str]
    claims: Optional[str] = None
    description: Optional[str] = None
    cited_by: Optional[List[str]] = None
    references: Optional[List[str]] = None
    url: str = ""
    
    def __post_init__(self):
        if self.cited_by is None:
            self.cited_by = []
        if self.references is None:
            self.references = []
        if not self.url:
            self.url = f"https://patents.google.com/patent/{self.patent_id}"

@dataclass
class PatentSearchResult:
    """专利搜索结果"""
    query: str
    total_count: int
    retrieved_count: int
    patents: List[Patent]
    search_timestamp: str
    api_version: str = "1.0.0"

class PatentRetriever:
    """
    专利检索器 - 支持多个专利数据源
    """
    
    def __init__(self):
        self.name = "PatentRetriever"
        self.version = "1.0.0"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PatentRetriever/1.0)'
        })
        
        logger.info(f"初始化专利检索器 v{self.version}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    async def search_patents(self, 
                           query: str, 
                           max_results: int = 20,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> PatentSearchResult:
        """
        异步搜索专利
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PatentSearchResult对象
        """
        search_timestamp = datetime.now().isoformat()
        
        try:
            logger.info(f"搜索专利: {query} (最多 {max_results} 件)")
            
            # 在线程池中运行同步代码
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                patents = await loop.run_in_executor(
                    pool, self._search_google_patents, query, max_results, start_date, end_date
                )
            
            logger.info(f"成功获取 {len(patents)} 件专利")
            
            return PatentSearchResult(
                query=query,
                total_count=len(patents),
                retrieved_count=len(patents),
                patents=patents,
                search_timestamp=search_timestamp
            )
            
        except Exception as e:
            logger.error(f"专利搜索失败: {e}")
            return PatentSearchResult(
                query=query,
                total_count=0,
                retrieved_count=0,
                patents=[],
                search_timestamp=search_timestamp
            )
    
    def _search_google_patents(self, 
                             query: str, 
                             max_results: int,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> List[Patent]:
        """
        使用Google Patents搜索专利（模拟实现）
        注意：实际应用中应使用正式的API
        """
        try:
            # 构建搜索URL
            search_url = f"https://patents.google.com/?q={query}&oq={query}"
            if start_date:
                search_url += f"&after={start_date.replace('-', '')}"
            if end_date:
                search_url += f"&before={end_date.replace('-', '')}"
            
            # 这里是模拟数据，实际应该解析网页或使用API
            # 在生产环境中，建议使用USPTO API或Google Patents API
            patents = self._generate_mock_patents(query, max_results)
            
            return patents
            
        except Exception as e:
            logger.error(f"Google Patents搜索失败: {e}")
            return []
    
    def _generate_mock_patents(self, query: str, count: int) -> List[Patent]:
        """生成模拟专利数据（用于演示）"""
        patents = []
        base_year = 2020
        
        for i in range(min(count, 10)):
            patent = Patent(
                patent_id=f"US{10000000 + i}B2",
                title=f"Method and system for {query} using epigenetic regulation - Example {i+1}",
                abstract=f"This invention relates to methods for regulating {query} through epigenetic mechanisms. "
                        f"The invention provides novel approaches for targeting {query} in therapeutic applications, "
                        f"particularly in cancer treatment and genetic disorders.",
                inventors=[f"Inventor {j+1}" for j in range(3)],
                assignee=f"Biotech Company {i % 5 + 1}",
                filing_date=f"{base_year + i // 3}-{(i % 12) + 1:02d}-15",
                publication_date=f"{base_year + i // 3 + 1}-{(i % 12) + 1:02d}-20",
                patent_type="Utility Patent",
                status="Active" if i < 5 else "Pending",
                classifications=["C12N", "A61K", "C07K"],
                claims=f"1. A method for modulating {query} expression comprising...\n"
                      f"2. The method of claim 1, wherein...\n"
                      f"3. A pharmaceutical composition comprising...",
                cited_by=[f"US{10001000 + j}B2" for j in range(2)],
                references=[f"US{9999000 + j}B2" for j in range(3)]
            )
            patents.append(patent)
        
        return patents
    
    async def search_by_gene(self, 
                           gene: str, 
                           additional_terms: Optional[List[str]] = None,
                           max_results: int = 20,
                           focus_areas: Optional[List[str]] = None) -> PatentSearchResult:
        """
        按基因名称搜索相关专利
        
        Args:
            gene: 基因名称
            additional_terms: 额外搜索词
            max_results: 最大结果数
            focus_areas: 关注领域 (如 ["therapy", "diagnostic", "CRISPR"])
        
        Returns:
            PatentSearchResult对象
        """
        # 构建专利搜索查询
        query_parts = [gene]
        
        # 添加表观遗传相关术语
        epigenetic_terms = ["epigenetic", "methylation", "histone", "chromatin"]
        query_parts.extend(epigenetic_terms[:2])  # 只添加前两个避免查询过于限制
        
        if additional_terms:
            query_parts.extend(additional_terms)
        
        if focus_areas:
            for area in focus_areas:
                if area.lower() == "therapy":
                    query_parts.append("(treatment OR therapeutic)")
                elif area.lower() == "diagnostic":
                    query_parts.append("(diagnostic OR detection)")
                elif area.lower() == "crispr":
                    query_parts.append("(CRISPR OR gene editing)")
        
        # 构建查询字符串
        query = " AND ".join(query_parts)
        
        return await self.search_patents(query, max_results)
    
    async def analyze_patent_landscape(self, patents: List[Patent]) -> Dict[str, Any]:
        """
        分析专利景观
        
        Args:
            patents: 专利列表
        
        Returns:
            分析结果字典
        """
        if not patents:
            return {
                "total_patents": 0,
                "message": "没有专利数据可供分析"
            }
        
        # 按年份统计
        year_distribution = {}
        for patent in patents:
            year = patent.filing_date.split('-')[0]
            year_distribution[year] = year_distribution.get(year, 0) + 1
        
        # 按受让人统计
        assignee_distribution = {}
        for patent in patents:
            assignee_distribution[patent.assignee] = assignee_distribution.get(patent.assignee, 0) + 1
        
        # 按状态统计
        status_distribution = {}
        for patent in patents:
            status_distribution[patent.status] = status_distribution.get(patent.status, 0) + 1
        
        # 技术分类统计
        classification_stats = {}
        for patent in patents:
            for cls in patent.classifications:
                classification_stats[cls] = classification_stats.get(cls, 0) + 1
        
        return {
            "total_patents": len(patents),
            "year_distribution": year_distribution,
            "top_assignees": dict(sorted(assignee_distribution.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]),
            "status_distribution": status_distribution,
            "top_classifications": dict(sorted(classification_stats.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]),
            "active_patents": sum(1 for p in patents if p.status == "Active"),
            "pending_patents": sum(1 for p in patents if p.status == "Pending")
        }


# 测试函数
async def test_patent_retriever():
    """测试专利检索器"""
    print("🧪 测试专利检索器")
    print("=" * 50)
    
    async with PatentRetriever() as retriever:
        # 测试基因相关专利搜索
        print("\n1. 测试基因专利搜索:")
        try:
            result = await retriever.search_by_gene(
                "BRCA1", 
                additional_terms=["cancer", "inhibitor"],
                max_results=5,
                focus_areas=["therapy", "diagnostic"]
            )
            print(f"   ✅ 搜索成功: {result.retrieved_count} 件专利")
            
            for i, patent in enumerate(result.patents[:3], 1):
                print(f"\n   专利 {i}:")
                print(f"   ID: {patent.patent_id}")
                print(f"   标题: {patent.title}")
                print(f"   受让人: {patent.assignee}")
                print(f"   状态: {patent.status}")
                print(f"   申请日期: {patent.filing_date}")
                
        except Exception as e:
            print(f"   ❌ 搜索失败: {e}")
        
        # 测试专利景观分析
        print("\n2. 测试专利景观分析:")
        if result and result.patents:
            try:
                landscape = await retriever.analyze_patent_landscape(result.patents)
                print(f"   ✅ 分析成功:")
                print(f"   总专利数: {landscape['total_patents']}")
                print(f"   活跃专利: {landscape['active_patents']}")
                print(f"   待审专利: {landscape['pending_patents']}")
                print(f"   主要受让人: {list(landscape['top_assignees'].keys())[:3]}")
            except Exception as e:
                print(f"   ❌ 分析失败: {e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_patent_retriever())
#!/usr/bin/env python3
"""
🧬 Enhanced Literature Expert - 增强文献分析专家
支持基因名、关键词、专业术语等多种查询方式
"""

import sys
import os
import asyncio
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# 核心依赖
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
except ImportError as e:
    print(f"❌ 缺少必要依赖，请安装: pip install sentence-transformers faiss-cpu")
    raise e

from Bio import Entrez
import xml.etree.ElementTree as ET

# 项目内部导入
from agent_core.clients.llm_client import call_llm
from agent_core.config.analysis_config import AnalysisConfig, AnalysisMode, ConfigManager

# 导入基础类（重用现有的数据结构）
from agent_core.agents.specialists.literature_expert import (
    LiteratureDocument, TextChunk, VectorStore, RAGProcessor, 
    CacheManager, SmartChunker, LiteratureAnalysisResult
)

logger = logging.getLogger(__name__)

# ===== 查询类型枚举 =====

from enum import Enum

class QueryType(Enum):
    """查询类型"""
    GENE = "gene"                    # 基因查询
    KEYWORD = "keyword"              # 关键词查询
    PROTEIN_FAMILY = "protein_family" # 蛋白家族查询
    MECHANISM = "mechanism"          # 机制查询
    COMPLEX = "complex"              # 复合查询

@dataclass
class SearchQuery:
    """搜索查询结构"""
    query_text: str                 # 查询文本
    query_type: QueryType           # 查询类型
    additional_terms: List[str] = None  # 附加术语
    exclude_terms: List[str] = None     # 排除术语
    date_range: tuple = None            # 日期范围 (start_year, end_year)
    max_results: int = 500              # 最大结果数
    
    def __post_init__(self):
        if self.additional_terms is None:
            self.additional_terms = []
        if self.exclude_terms is None:
            self.exclude_terms = []

# ===== 增强的PubMed检索器 =====

class EnhancedPubMedRetriever:
    """增强的PubMed文献检索器 - 支持多种查询类型"""
    
    def __init__(self):
        self.name = "Enhanced PubMed Retriever"
        self.version = "3.0.0"
        # 配置Bio.Entrez
        Entrez.email = "czqrainy@gmail.com"
        Entrez.api_key = "983222f9d5a2a81facd7d158791d933e6408"
        
        # 预定义的搜索模板
        self.search_templates = {
            QueryType.GENE: [
                "{query}[Title/Abstract]",
                '"{query}" AND (disease OR treatment OR therapy)',
                "{query} AND (clinical trial[Publication Type] OR clinical study[Publication Type])",
                "{query} AND (mechanism OR pathway OR function)",
                "{query} AND (drug OR inhibitor OR target OR therapeutic)"
            ],
            QueryType.KEYWORD: [
                "{query}[Title/Abstract]",
                '"{query}" AND (regulation OR expression OR function)',
                "{query} AND (signaling OR pathway OR mechanism)",
                "{query} AND (therapeutic OR treatment OR clinical)",
                "{query} AND (protein OR gene OR molecular)"
            ],
            QueryType.PROTEIN_FAMILY: [
                '"{query}" AND (protein OR family OR domain)',
                "{query} AND (structure OR function OR binding)",
                "{query} AND (regulation OR expression OR localization)",
                "{query} AND (interaction OR complex OR assembly)",
                "{query} AND (evolution OR conservation OR phylogeny)"
            ],
            QueryType.MECHANISM: [
                '"{query}" AND (mechanism OR pathway OR process)',
                "{query} AND (regulation OR control OR modulation)",
                "{query} AND (signaling OR cascade OR network)",
                "{query} AND (molecular OR cellular OR biological)",
                "{query} AND (function OR role OR activity)"
            ],
            QueryType.COMPLEX: [
                "{query}",  # 复合查询直接使用原始查询
                '"{query}" AND review[Publication Type]',
                "{query} AND recent[Filter]"
            ]
        }
    
    async def search_literature(self, search_query: Union[str, SearchQuery], max_results: int = 500) -> List[LiteratureDocument]:
        """
        检索文献 - 支持多种查询类型
        
        Args:
            search_query: 查询字符串或SearchQuery对象
            max_results: 最大结果数
        
        Returns:
            文献文档列表
        """
        
        # 处理输入参数
        if isinstance(search_query, str):
            # 兼容原有接口：字符串查询默认为基因查询
            query = SearchQuery(
                query_text=search_query,
                query_type=QueryType.GENE,
                max_results=max_results
            )
        else:
            query = search_query
            max_results = query.max_results
        
        print(f"📚 检索文献: {query.query_text} ({query.query_type.value})")
        print(f"   目标: {max_results} 篇")
        
        # 构建搜索策略
        search_strategies = self._build_search_strategies(query)
        
        all_documents = []
        seen_pmids = set()
        
        for i, strategy in enumerate(search_strategies):
            print(f"  🔍 搜索策略 {i+1}: {strategy}")
            
            try:
                docs = await self._execute_search(strategy, max_results // len(search_strategies))
                
                for doc in docs:
                    if doc.pmid not in seen_pmids:
                        seen_pmids.add(doc.pmid)
                        all_documents.append(doc)
                        
                        if len(all_documents) >= max_results:
                            break
                
                print(f"    ✅ 新增 {len(docs)} 篇，累计 {len(all_documents)} 篇")
                
                if len(all_documents) >= max_results:
                    break
                    
            except Exception as e:
                print(f"    ❌ 搜索失败: {e}")
                continue
        
        print(f"📊 检索完成: 共 {len(all_documents)} 篇文献")
        return all_documents[:max_results]
    
    def _build_search_strategies(self, query: SearchQuery) -> List[str]:
        """构建搜索策略"""
        
        base_strategies = self.search_templates.get(query.query_type, self.search_templates[QueryType.KEYWORD])
        strategies = []
        
        # 基础查询策略
        for template in base_strategies:
            strategy = template.format(query=query.query_text)
            strategies.append(strategy)
        
        # 添加附加术语
        if query.additional_terms:
            additional_query = f"({query.query_text}) AND ({' OR '.join(query.additional_terms)})"
            strategies.append(additional_query)
        
        # 处理排除术语
        if query.exclude_terms:
            exclude_part = " AND ".join([f"NOT {term}" for term in query.exclude_terms])
            enhanced_strategies = []
            for strategy in strategies[:2]:  # 只对前两个策略应用排除
                enhanced_strategies.append(f"{strategy} {exclude_part}")
            strategies.extend(enhanced_strategies)
        
        # 日期范围过滤
        if query.date_range:
            start_year, end_year = query.date_range
            date_filter = f" AND {start_year}[PDAT]:{end_year}[PDAT]"
            dated_strategies = []
            for strategy in strategies[:3]:  # 对前三个策略应用日期过滤
                dated_strategies.append(f"{strategy}{date_filter}")
            strategies.extend(dated_strategies)
        
        return strategies
    
    async def _execute_search(self, query: str, max_results: int) -> List[LiteratureDocument]:
        """执行单次搜索"""
        
        try:
            # 1. 搜索PMID
            search_handle = Entrez.esearch(
                db="pubmed", 
                term=query, 
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            pmid_list = search_results["IdList"]
            
            if not pmid_list:
                return []
            
            # 2. 批量获取详情
            documents = []
            batch_size = 50
            
            for i in range(0, len(pmid_list), batch_size):
                batch_pmids = pmid_list[i:i+batch_size]
                batch_docs = await self._fetch_batch_details(batch_pmids)
                documents.extend(batch_docs)
                
                # API限流
                if i + batch_size < len(pmid_list):
                    await asyncio.sleep(0.5)
            
            return documents
            
        except Exception as e:
            print(f"❌ 搜索执行失败: {e}")
            return []
    
    async def _fetch_batch_details(self, pmid_list: List[str]) -> List[LiteratureDocument]:
        """批量获取文献详情"""
        
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmid_list),
                rettype="medline",
                retmode="xml"
            )
            
            xml_data = fetch_handle.read()
            fetch_handle.close()
            
            # 解析XML
            root = ET.fromstring(xml_data)
            articles = root.findall(".//PubmedArticle")
            
            documents = []
            for article_xml in articles:
                doc = self._parse_article(article_xml)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return []
    
    def _parse_article(self, article_xml) -> Optional[LiteratureDocument]:
        """解析单篇文章"""
        
        try:
            # 基本信息
            pmid = article_xml.findtext(".//PMID", "")
            title = article_xml.findtext(".//ArticleTitle", "")
            
            # 摘要处理
            abstract_elem = article_xml.find(".//Abstract")
            abstract = ""
            if abstract_elem is not None:
                abstract_texts = []
                for text_elem in abstract_elem.findall(".//AbstractText"):
                    text = text_elem.text or ""
                    label = text_elem.get("Label", "")
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
                abstract = " ".join(abstract_texts)
            
            # 作者
            authors = []
            for author in article_xml.findall(".//Author"):
                last_name = author.findtext("LastName", "")
                first_name = author.findtext("ForeName", "")
                if last_name:
                    authors.append(f"{first_name} {last_name}".strip())
            
            # 期刊和年份
            journal = article_xml.findtext(".//Journal/Title", "")
            year_elem = article_xml.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None and year_elem.text else 0
            
            # DOI
            doi = ""
            for article_id in article_xml.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text or ""
                    break
            
            if not title or not abstract:
                return None
            
            return LiteratureDocument(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                doi=doi
            )
            
        except Exception as e:
            return None

# ===== 增强的文献分析专家 =====

class EnhancedLiteratureExpert:
    """增强文献分析专家 - 支持多种查询方式"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.name = "Enhanced Literature Expert"
        self.version = "3.0.0"
        self.expertise = ["多类型查询", "文献分析", "机制研究", "治疗策略", "靶点分析"]
        
        # 配置
        self.config = config or ConfigManager.get_standard_config()
        
        # 组件
        self.retriever = EnhancedPubMedRetriever()
        self.chunker = SmartChunker(chunk_size=250, overlap=50)
        self.cache_manager = EnhancedCacheManager()
        
        logger.info(f"Enhanced Literature Expert 初始化完成 - {self.version}")
    
    def set_config(self, config: AnalysisConfig):
        """设置配置"""
        self.config = config
        logger.info(f"配置已更新: {config.mode.value}")
    
    def set_mode(self, mode: AnalysisMode):
        """设置模式"""
        self.config = ConfigManager.get_config_by_mode(mode)
        logger.info(f"模式切换: {mode.value}")
    
    async def analyze_by_gene(self, gene_target: str, context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """基因名分析（兼容原有接口）"""
        
        query = SearchQuery(
            query_text=gene_target,
            query_type=QueryType.GENE,
            max_results=self._get_max_literature()
        )
        
        return await self.analyze_by_query(query, context)
    
    async def analyze_by_keyword(self, keyword: str, 
                                additional_terms: List[str] = None,
                                exclude_terms: List[str] = None,
                                context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """关键词分析"""
        
        query = SearchQuery(
            query_text=keyword,
            query_type=QueryType.KEYWORD,
            additional_terms=additional_terms or [],
            exclude_terms=exclude_terms or [],
            max_results=self._get_max_literature()
        )
        
        return await self.analyze_by_query(query, context)
    
    async def analyze_protein_family(self, family_name: str,
                                   additional_terms: List[str] = None,
                                   context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """蛋白家族分析"""
        
        query = SearchQuery(
            query_text=family_name,
            query_type=QueryType.PROTEIN_FAMILY,
            additional_terms=additional_terms or [],
            max_results=self._get_max_literature()
        )
        
        return await self.analyze_by_query(query, context)
    
    async def analyze_mechanism(self, mechanism_query: str,
                              additional_terms: List[str] = None,
                              context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """机制分析"""
        
        query = SearchQuery(
            query_text=mechanism_query,
            query_type=QueryType.MECHANISM,
            additional_terms=additional_terms or [],
            max_results=self._get_max_literature()
        )
        
        return await self.analyze_by_query(query, context)
    
    async def analyze_by_query(self, search_query: SearchQuery, context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """
        通用查询分析方法
        
        Args:
            search_query: 搜索查询对象
            context: 上下文配置
        
        Returns:
            文献分析结果
        """
        
        logger.info(f"开始文献分析: {search_query.query_text} ({search_query.query_type.value}) - 模式: {self.config.mode.value}")
        
        try:
            # 确定分析参数
            top_k = self._get_top_k()
            
            # 1. 尝试从缓存加载
            cache_key = self._generate_cache_key(search_query)
            vector_store = self.cache_manager.load_by_key(cache_key)
            
            # 2. 如果缓存无效，重新构建
            if vector_store is None:
                vector_store = await self._build_literature_index(search_query)
                # 保存缓存
                self.cache_manager.save_by_key(cache_key, vector_store)
            
            # 3. RAG查询处理
            rag_processor = RAGProcessor(vector_store)
            
            print("🤖 开始RAG查询...")
            
            # 根据查询类型调整RAG查询
            rag_queries = self._get_rag_queries(search_query)
            
            # 并发处理查询
            tasks = [
                rag_processor.process_query(search_query.query_text, query_type, top_k)
                for query_type in rag_queries
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 4. 构建分析结果
            references = self._extract_references(vector_store.chunks)
            confidence_score = self._calculate_confidence(vector_store.chunks)
            
            # 根据查询类型组织结果
            result_dict = {}
            for i, query_type in enumerate(rag_queries):
                result_dict[query_type] = results[i] if i < len(results) else ""
            
            analysis_result = LiteratureAnalysisResult(
                gene_target=search_query.query_text,  # 保持兼容性
                disease_mechanism=result_dict.get("disease_mechanism", ""),
                treatment_strategy=result_dict.get("treatment_strategy", ""),
                target_analysis=result_dict.get("target_analysis", ""),
                references=references[:50],  # 限制引用数量
                total_literature=len(set(chunk.doc_id for chunk in vector_store.chunks)),
                total_chunks=len(vector_store.chunks),
                confidence_score=confidence_score,
                analysis_method=f"Enhanced-RAG-{search_query.query_type.value}",
                timestamp=datetime.now().isoformat(),
                config_used=self._get_config_summary(),
                token_usage=self._estimate_token_usage(top_k)
            )
            
            logger.info(f"文献分析完成: {search_query.query_text} - 文献数: {analysis_result.total_literature}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"文献分析失败: {search_query.query_text} - {str(e)}")
            return self._create_error_result(search_query.query_text, str(e))
    
    def _get_rag_queries(self, search_query: SearchQuery) -> List[str]:
        """根据查询类型获取RAG查询类型"""
        
        if search_query.query_type == QueryType.GENE:
            return ["disease_mechanism", "treatment_strategy", "target_analysis"]
        elif search_query.query_type == QueryType.KEYWORD:
            # 使用兼容的查询类型
            return ["disease_mechanism", "treatment_strategy", "target_analysis"]
        elif search_query.query_type == QueryType.PROTEIN_FAMILY:
            # 使用兼容的查询类型
            return ["disease_mechanism", "treatment_strategy", "target_analysis"]
        elif search_query.query_type == QueryType.MECHANISM:
            # 使用兼容的查询类型
            return ["disease_mechanism", "treatment_strategy", "target_analysis"]
        else:
            return ["disease_mechanism", "treatment_strategy", "target_analysis"]
    
    async def _build_literature_index(self, search_query: SearchQuery) -> VectorStore:
        """构建文献索引"""
        
        print(f"🏗️ 构建文献索引: {search_query.query_text} ({search_query.query_type.value})")
        
        # 1. 检索文献
        documents = await self.retriever.search_literature(search_query)
        
        if not documents:
            raise ValueError(f"未找到 {search_query.query_text} 相关文献")
        
        # 2. 文本分块
        chunks = self.chunker.chunk_documents(documents)
        
        # 3. 构建向量索引
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        
        return vector_store
    
    def _generate_cache_key(self, search_query: SearchQuery) -> str:
        """生成缓存键"""
        
        query_str = f"{search_query.query_text}_{search_query.query_type.value}"
        if search_query.additional_terms:
            query_str += f"_add_{','.join(search_query.additional_terms)}"
        if search_query.exclude_terms:
            query_str += f"_exc_{','.join(search_query.exclude_terms)}"
        if search_query.date_range:
            query_str += f"_date_{search_query.date_range[0]}_{search_query.date_range[1]}"
        
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _get_max_literature(self) -> int:
        """获取最大文献数量"""
        if self.config.mode == AnalysisMode.QUICK:
            return 100
        elif self.config.mode == AnalysisMode.STANDARD:
            return 500
        elif self.config.mode == AnalysisMode.DEEP:
            return 1000
        else:
            return 500
    
    def _get_top_k(self) -> int:
        """获取top-k参数"""
        if self.config.mode == AnalysisMode.QUICK:
            return 10
        elif self.config.mode == AnalysisMode.STANDARD:
            return 15
        elif self.config.mode == AnalysisMode.DEEP:
            return 25
        else:
            return 15
    
    def _extract_references(self, chunks: List[TextChunk]) -> List[Dict]:
        """提取引用信息"""
        
        references = {}
        for chunk in chunks:
            pmid = chunk.metadata.get("pmid", "")
            if pmid and pmid not in references:
                references[pmid] = {
                    "PMID": pmid,
                    "Title": chunk.metadata.get("title", ""),
                    "Journal": chunk.metadata.get("journal", ""),
                    "Year": chunk.metadata.get("year", 0)
                }
        
        return list(references.values())
    
    def _calculate_confidence(self, chunks: List[TextChunk]) -> float:
        """计算置信度"""
        
        if not chunks:
            return 0.0
        
        # 基于文献数量、质量等因素计算置信度
        unique_docs = len(set(chunk.doc_id for chunk in chunks))
        recent_docs = sum(1 for chunk in chunks if chunk.metadata.get("year", 0) >= 2020)
        
        base_confidence = min(unique_docs / 100, 1.0)  # 基础置信度
        recency_bonus = (recent_docs / unique_docs) * 0.2 if unique_docs > 0 else 0  # 时效性加分
        
        return min(base_confidence + recency_bonus, 1.0)
    
    def _get_config_summary(self) -> Dict:
        """获取配置摘要"""
        
        return {
            "mode": self.config.mode.value,
            "max_literature": self._get_max_literature(),
            "top_k": self._get_top_k()
        }
    
    def _estimate_token_usage(self, top_k: int) -> Dict:
        """估算Token使用量"""
        
        return {
            "estimated_input_tokens": top_k * 200,  # 每个chunk约200 tokens
            "estimated_output_tokens": 1500,       # 输出约1500 tokens
            "total_estimated": top_k * 200 + 1500
        }
    
    def _create_error_result(self, query_text: str, error_msg: str) -> LiteratureAnalysisResult:
        """创建错误结果"""
        
        return LiteratureAnalysisResult(
            gene_target=query_text,
            disease_mechanism=f"分析失败: {error_msg}",
            treatment_strategy="",
            target_analysis="",
            references=[],
            total_literature=0,
            total_chunks=0,
            confidence_score=0.0,
            analysis_method="error",
            timestamp=datetime.now().isoformat(),
            config_used={},
            token_usage={}
        )

# ===== 缓存管理器扩展 =====

class EnhancedCacheManager:
    """增强的缓存管理器"""
    
    def __init__(self, cache_dir: str = "enhanced_literature_cache"):
        self.cache_dir = cache_dir
        self.cache_days = 7  # 缓存有效期
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_by_key(self, cache_key: str) -> Optional[VectorStore]:
        """根据缓存键加载"""
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # 检查缓存时效
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mod_time > timedelta(days=self.cache_days):
                return None
            
            with open(cache_file, 'rb') as f:
                vector_store = pickle.load(f)
                print(f"📂 从缓存加载: {cache_key}")
                return vector_store
                
        except Exception as e:
            print(f"❌ 缓存加载失败: {e}")
            return None
    
    def save_by_key(self, cache_key: str, vector_store: VectorStore):
        """根据缓存键保存"""
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(vector_store, f)
            print(f"💾 缓存已保存: {cache_key}")
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
    
    # 兼容原有接口
    def load(self, gene: str, max_results: int) -> Optional[VectorStore]:
        """兼容原有load方法"""
        cache_key = f"{gene}_{max_results}"
        return self.load_by_key(cache_key)
    
    def save(self, gene: str, max_results: int, vector_store: VectorStore):
        """兼容原有save方法"""
        cache_key = f"{gene}_{max_results}"
        self.save_by_key(cache_key, vector_store)

# ===== 使用示例和测试函数 =====

async def test_enhanced_literature_expert():
    """测试增强文献专家"""
    
    expert = EnhancedLiteratureExpert()
    
    print("🧪 测试1: KRAB-like 蛋白查询")
    result1 = await expert.analyze_by_keyword(
        keyword="krab-like",
        additional_terms=["epigenetic", "transcriptional regulation", "zinc finger"],
        exclude_terms=["virus", "bacterial"]
    )
    
    print(f"结果: {result1.total_literature} 篇文献")
    
    print("\n🧪 测试2: 蛋白家族查询")
    result2 = await expert.analyze_protein_family(
        family_name="KRAB domain proteins",
        additional_terms=["chromatin modification", "gene silencing"]
    )
    
    print(f"结果: {result2.total_literature} 篇文献")
    
    print("\n🧪 测试3: 机制查询")
    result3 = await expert.analyze_mechanism(
        mechanism_query="epigenetic regulation by zinc finger proteins",
        additional_terms=["DNA methylation", "histone modification"]
    )
    
    print(f"结果: {result3.total_literature} 篇文献")
    
    return [result1, result2, result3]

if __name__ == "__main__":
    asyncio.run(test_enhanced_literature_expert())
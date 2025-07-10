# agent_core/agents/specialists/literature_expert.py - 基于RAG的文献分析专家

"""
🧬 Literature Expert - 文献分析专家
支持RAG优化的大规模文献分析，大幅节省Token消耗
"""

import sys
import os
import asyncio
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional
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

logger = logging.getLogger(__name__)

# ===== 数据结构定义 =====

@dataclass
class LiteratureDocument:
    """文献文档结构"""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = None
    journal: str = ""
    year: int = 0
    doi: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
    
    def to_text(self) -> str:
        """转换为可搜索的文本"""
        return f"标题: {self.title}\n摘要: {self.abstract}"

@dataclass
class TextChunk:
    """文本块结构"""
    text: str
    doc_id: str  # PMID
    chunk_id: str
    metadata: Dict
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(f"{self.doc_id}_{self.text[:50]}".encode()).hexdigest()[:12]

@dataclass
class LiteratureAnalysisResult:
    """文献分析结果"""
    gene_target: str
    disease_mechanism: str
    treatment_strategy: str
    target_analysis: str
    references: List[Dict]
    total_literature: int
    total_chunks: int
    confidence_score: float
    analysis_method: str
    timestamp: str
    config_used: Dict
    token_usage: Dict

# ===== PubMed检索器 =====

class PubMedRetriever:
    """PubMed文献检索器"""
    
    def __init__(self):
        self.name = "PubMed Retriever"
        self.version = "2.0.0"
        # 配置Bio.Entrez
        Entrez.email = "czqrainy@gmail.com"
        Entrez.api_key = "983222f9d5a2a81facd7d158791d933e6408"
    
    async def search_literature(self, gene: str, max_results: int = 500) -> List[LiteratureDocument]:
        """检索文献"""
        
        print(f"📚 检索 {gene} 相关文献，目标: {max_results} 篇")
        
        # 多策略搜索
        search_strategies = [
            f"{gene}[Title/Abstract]",
            f'"{gene}" AND (disease OR treatment OR therapy)',
            f"{gene} AND (clinical trial[Publication Type] OR clinical study[Publication Type])",
            f"{gene} AND (mechanism OR pathway OR function)",
            f"{gene} AND (drug OR inhibitor OR target OR therapeutic)"
        ]
        
        all_documents = []
        seen_pmids = set()
        
        for strategy in search_strategies:
            print(f"  🔍 搜索策略: {strategy}")
            
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
            
            root = ET.fromstring(fetch_handle.read())
            
            documents = []
            for article in root.findall(".//PubmedArticle"):
                doc = self._parse_article(article)
                if doc and doc.abstract:  # 只保留有摘要的
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

# ===== 文本分块器 =====

class SmartChunker:
    """智能文本分块器"""
    
    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, documents: List[LiteratureDocument]) -> List[TextChunk]:
        """分块文档"""
        
        print(f"📝 开始文本分块，块大小: {self.chunk_size}")
        
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_single_doc(doc)
            all_chunks.extend(chunks)
        
        print(f"✅ 分块完成: {len(documents)} 篇 → {len(all_chunks)} 块")
        return all_chunks
    
    def _chunk_single_doc(self, doc: LiteratureDocument) -> List[TextChunk]:
        """分块单个文档"""
        
        chunks = []
        
        # 1. 标题块（重要）
        title_chunk = TextChunk(
            text=f"标题: {doc.title}",
            doc_id=doc.pmid,
            chunk_id=f"{doc.pmid}_title",
            metadata={
                "pmid": doc.pmid,
                "title": doc.title,
                "journal": doc.journal,
                "year": doc.year,
                "chunk_type": "title"
            }
        )
        chunks.append(title_chunk)
        
        # 2. 摘要分块
        abstract_chunks = self._chunk_abstract(doc)
        chunks.extend(abstract_chunks)
        
        return chunks
    
    def _chunk_abstract(self, doc: LiteratureDocument) -> List[TextChunk]:
        """分块摘要"""
        
        abstract = doc.abstract
        if len(abstract) <= self.chunk_size:
            # 短摘要，整体作为一块
            return [TextChunk(
                text=f"摘要: {abstract}",
                doc_id=doc.pmid,
                chunk_id=f"{doc.pmid}_abstract",
                metadata={
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "journal": doc.journal,
                    "year": doc.year,
                    "chunk_type": "abstract"
                }
            )]
        
        # 长摘要，按句子分块
        sentences = self._split_sentences(abstract)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(TextChunk(
                        text=f"摘要: {current_chunk}",
                        doc_id=doc.pmid,
                        chunk_id=f"{doc.pmid}_abstract_{chunk_index}",
                        metadata={
                            "pmid": doc.pmid,
                            "title": doc.title,
                            "journal": doc.journal,
                            "year": doc.year,
                            "chunk_type": "abstract_part",
                            "part_index": chunk_index
                        }
                    ))
                    chunk_index += 1
                
                current_chunk = sentence
        
        # 最后一块
        if current_chunk:
            chunks.append(TextChunk(
                text=f"摘要: {current_chunk}",
                doc_id=doc.pmid,
                chunk_id=f"{doc.pmid}_abstract_{chunk_index}",
                metadata={
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "journal": doc.journal,
                    "year": doc.year,
                    "chunk_type": "abstract_part",
                    "part_index": chunk_index
                }
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        
        sentences = []
        current = ""
        
        for i, char in enumerate(text):
            current += char
            if char in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n':
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if len(s) > 10]

# ===== 向量存储系统 =====

class VectorStore:
    """向量存储和检索"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def build_index(self, chunks: List[TextChunk]):
        """构建向量索引"""
        
        print(f"🔍 构建向量索引，模型: {self.model_name}")
        
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        print(f"  📊 编码 {len(texts)} 个文本块...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # 构建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # 标准化用于余弦相似度
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ 索引构建完成: {len(chunks)} 块, 维度: {dimension}")
    
    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        """搜索相关块"""
        
        if self.index is None:
            raise ValueError("索引未构建")
        
        # 编码查询
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "chunk": chunk,
                    "score": float(score),
                    "text": chunk.text,
                    "metadata": chunk.metadata
                })
        
        return results
    
    def save(self, file_path: str):
        """保存索引"""
        
        save_data = {
            "chunks": self.chunks,
            "model_name": self.model_name
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        faiss.write_index(self.index, file_path + ".faiss")
        print(f"💾 索引已保存: {file_path}")
    
    def load(self, file_path: str) -> bool:
        """加载索引"""
        
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.chunks = save_data["chunks"]
            self.model_name = save_data["model_name"]
            self.index = faiss.read_index(file_path + ".faiss")
            
            print(f"📂 索引已加载: {len(self.chunks)} 块")
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

# ===== RAG查询处理器 =====

class RAGProcessor:
    """RAG查询处理器"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.query_templates = {
            "disease_mechanism": "该基因与哪些疾病相关？疾病的发病机制是什么？有什么临床需求？",
            "treatment_strategy": "有哪些治疗方法和策略？包括药物、疗法等？临床研究现状如何？",
            "target_analysis": "该基因的作用通路是什么？有哪些潜在治疗靶点？研究进展如何？"
        }
    
    async def process_query(self, gene: str, query_type: str, top_k: int = 15) -> str:
        """处理RAG查询"""
        
        print(f"🤖 RAG查询: {gene} - {query_type}")
        
        # 构建查询
        base_query = self.query_templates.get(query_type, "")
        full_query = f"{gene} {base_query}"
        
        # 检索相关块
        relevant_chunks = self.vector_store.search(full_query, top_k)
        
        if not relevant_chunks:
            return f"未找到与 {gene} 相关的 {query_type} 信息。"
        
        print(f"  📊 检索到 {len(relevant_chunks)} 个相关块")
        
        # 构建prompt
        prompt = self._build_prompt(gene, query_type, relevant_chunks)
        
        # LLM生成
        response = call_llm(prompt)
        return response
    
    def _build_prompt(self, gene: str, query_type: str, relevant_chunks: List[Dict]) -> str:
        """构建RAG prompt"""
        
        # 整理上下文
        context_blocks = []
        for i, chunk_info in enumerate(relevant_chunks):
            chunk = chunk_info["chunk"]
            score = chunk_info["score"]
            pmid = chunk.metadata.get("pmid", "")
            
            context_blocks.append(
                f"[{i+1}] (PMID: {pmid}, 相似度: {score:.3f})\n{chunk.text}"
            )
        
        context_text = "\n\n".join(context_blocks)
        
        # 根据查询类型构建prompt
        if query_type == "disease_mechanism":
            prompt = f"""你是资深医学专家，请基于以下文献信息回答关于基因 {gene} 的问题。

问题：该基因涉及哪些疾病？这些疾病的发病机制是怎样的？有哪些尚未满足的临床需求？

请仔细阅读以下相关文献段落，并基于这些信息进行回答。在引用具体信息时，请使用 [1]、[2] 等标记。

相关文献段落：
{context_text}

请以如下结构输出：
### 疾病机制与临床需求（Gene: {gene}）
- 疾病关联：
- 发病机制：
- 临床需求：

注意：只基于提供的文献信息回答，不要添加未提及的内容。"""

        elif query_type == "treatment_strategy":
            prompt = f"""你是医学临床顾问，请基于以下文献信息分析基因 {gene} 相关的治疗策略。

问题：当前与该基因相关的治疗方法有哪些？包括传统治疗、靶向药物、免疫治疗等。

请仔细阅读以下相关文献段落，并基于这些信息进行回答。在引用具体信息时，请使用 [1]、[2] 等标记。

相关文献段落：
{context_text}

请以如下结构输出：
### 治疗策略分析（Gene: {gene}）
- 已有治疗策略：
- 临床研究现状：
- 与该基因直接相关的干预方法：

注意：只基于提供的文献信息回答，不要添加未提及的内容。"""

        elif query_type == "target_analysis":
            prompt = f"""你是药物研发专家，请基于以下文献信息分析基因 {gene} 的靶点潜力。

问题：与该基因相关的关键通路、蛋白复合物或信号机制？有哪些潜在干预位点值得关注？研究处于何种阶段？

请仔细阅读以下相关文献段落，并基于这些信息进行回答。在引用具体信息时，请使用 [1]、[2] 等标记。

相关文献段落：
{context_text}

请以如下结构输出：
### 靶点分析与研究进展（Gene: {gene}）
- 作用通路：
- 潜在靶点：
- 研究状态：

注意：只基于提供的文献信息回答，不要添加未提及的内容。"""

        return prompt

# ===== 缓存管理器 =====

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "literature_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, gene: str, max_results: int) -> str:
        """获取缓存路径"""
        cache_key = f"{gene}_{max_results}"
        return os.path.join(self.cache_dir, f"{cache_key}")
    
    def is_valid(self, cache_path: str, max_age_days: int = 7) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time < timedelta(days=max_age_days)
    
    def save(self, gene: str, max_results: int, vector_store: VectorStore):
        """保存缓存"""
        cache_path = self.get_cache_path(gene, max_results)
        vector_store.save(cache_path)
    
    def load(self, gene: str, max_results: int) -> Optional[VectorStore]:
        """加载缓存"""
        cache_path = self.get_cache_path(gene, max_results)
        
        if self.is_valid(cache_path):
            vector_store = VectorStore()
            if vector_store.load(cache_path):
                print(f"📂 从缓存加载: {gene}")
                return vector_store
        
        return None

# ===== 主要的Literature Expert =====

class LiteratureExpert:
    """文献分析专家 - 基于RAG优化"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.name = "Literature Expert"
        self.version = "2.0.0"
        self.expertise = ["文献分析", "机制研究", "治疗策略", "靶点分析"]
        
        # 配置
        self.config = config or ConfigManager.get_standard_config()
        
        # 组件
        self.retriever = PubMedRetriever()
        self.chunker = SmartChunker(chunk_size=250, overlap=50)
        self.cache_manager = CacheManager()
        
        logger.info(f"Literature Expert 初始化完成 - {self.version}")
    
    def set_config(self, config: AnalysisConfig):
        """设置配置"""
        self.config = config
        logger.info(f"配置已更新: {config.mode.value}")
    
    def set_mode(self, mode: AnalysisMode):
        """设置模式"""
        self.config = ConfigManager.get_config_by_mode(mode)
        logger.info(f"模式切换: {mode.value}")
    
    async def analyze(self, gene_target: str, context: Dict[str, Any] = None) -> LiteratureAnalysisResult:
        """
        主要分析方法
        
        Args:
            gene_target: 目标基因
            context: 上下文配置
        
        Returns:
            文献分析结果
        """
        
        logger.info(f"开始文献分析: {gene_target} - 模式: {self.config.mode.value}")
        
        try:
            # 确定分析参数
            max_literature = self._get_max_literature()
            top_k = self._get_top_k()
            
            # 1. 尝试从缓存加载
            vector_store = self.cache_manager.load(gene_target, max_literature)
            
            # 2. 如果缓存无效，重新构建
            if vector_store is None:
                vector_store = await self._build_literature_index(gene_target, max_literature)
                # 保存缓存
                self.cache_manager.save(gene_target, max_literature, vector_store)
            
            # 3. RAG查询处理
            rag_processor = RAGProcessor(vector_store)
            
            print("🤖 开始RAG查询...")
            
            # 并发处理三个查询
            tasks = [
                rag_processor.process_query(gene_target, "disease_mechanism", top_k),
                rag_processor.process_query(gene_target, "treatment_strategy", top_k),
                rag_processor.process_query(gene_target, "target_analysis", top_k)
            ]
            
            results = await asyncio.gather(*tasks)
            disease_result, treatment_result, target_result = results
            
            # 4. 构建分析结果
            references = self._extract_references(vector_store.chunks)
            confidence_score = self._calculate_confidence(vector_store.chunks)
            
            analysis_result = LiteratureAnalysisResult(
                gene_target=gene_target,
                disease_mechanism=disease_result,
                treatment_strategy=treatment_result,
                target_analysis=target_result,
                references=references[:50],  # 限制引用数量
                total_literature=len(set(chunk.doc_id for chunk in vector_store.chunks)),
                total_chunks=len(vector_store.chunks),
                confidence_score=confidence_score,
                analysis_method="RAG-optimized",
                timestamp=datetime.now().isoformat(),
                config_used=self._get_config_summary(),
                token_usage=self._estimate_token_usage(top_k)
            )
            
            logger.info(f"文献分析完成: {gene_target} - 文献数: {analysis_result.total_literature}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"文献分析失败: {gene_target} - {str(e)}")
            return self._create_error_result(gene_target, str(e))
    
    async def _build_literature_index(self, gene: str, max_results: int) -> VectorStore:
        """构建文献索引"""
        
        print(f"🏗️ 构建文献索引: {gene}")
        
        # 1. 检索文献
        documents = await self.retriever.search_literature(gene, max_results)
        
        if not documents:
            raise ValueError(f"未找到 {gene} 相关文献")
        
        # 2. 文本分块
        chunks = self.chunker.chunk_documents(documents)
        
        # 3. 构建向量索引
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        
        return vector_store
    
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
        
        # 基于文献数量和质量的简单评分
        unique_docs = len(set(chunk.doc_id for chunk in chunks))
        
        if unique_docs >= 50:
            return 0.9
        elif unique_docs >= 20:
            return 0.8
        elif unique_docs >= 10:
            return 0.7
        elif unique_docs >= 5:
            return 0.6
        else:
            return 0.5
    
    def _estimate_token_usage(self, top_k: int) -> Dict:
        """估算Token使用"""
        
        # RAG方式的Token估算
        input_tokens = top_k * 200  # 每个相关块约200 tokens
        output_tokens = 1000 * 3   # 三个问题各1000 tokens输出
        total_tokens = input_tokens + output_tokens
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_tokens * 0.000002
        }
    
    def _get_config_summary(self) -> Dict:
        """获取配置摘要"""
        
        return {
            "mode": self.config.mode.value,
            "max_literature": self._get_max_literature(),
            "top_k": self._get_top_k(),
            "analysis_method": "RAG-optimized"
        }
    
    def _create_error_result(self, gene_target: str, error_msg: str) -> LiteratureAnalysisResult:
        """创建错误结果"""
        
        return LiteratureAnalysisResult(
            gene_target=gene_target,
            disease_mechanism=f"分析 {gene_target} 时发生错误: {error_msg}",
            treatment_strategy="",
            target_analysis="",
            references=[],
            total_literature=0,
            total_chunks=0,
            confidence_score=0.0,
            analysis_method="error",
            timestamp=datetime.now().isoformat(),
            config_used=self._get_config_summary(),
            token_usage={}
        )
    
    def export_results(self, result: LiteratureAnalysisResult, format: str = "dict") -> Any:
        """导出结果"""
        
        if format == "dict":
            return asdict(result)
        elif format == "json":
            import json
            return json.dumps(asdict(result), indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def estimate_analysis_cost(self, gene_target: str) -> Dict[str, Any]:
        """估算分析成本"""
        
        token_estimate = self._estimate_token_usage(self._get_top_k())
        
        return {
            "gene_target": gene_target,
            "estimated_tokens": token_estimate["total_tokens"],
            "estimated_cost_usd": token_estimate["estimated_cost_usd"],
            "estimated_time_seconds": 60,  # RAG分析约1分钟
            "config_mode": self.config.mode.value,
            "max_literature": self._get_max_literature()
        }
    
    def __str__(self) -> str:
        return f"LiteratureExpert(name='{self.name}', version='{self.version}', mode='{self.config.mode.value}')"

# ===== 使用示例 =====

async def example_usage():
    """使用示例"""
    
    print("🧬 Literature Expert 使用示例")
    print("=" * 60)
    
    # 1. 创建Literature Expert
    expert = LiteratureExpert()
    expert.set_mode(AnalysisMode.STANDARD)
    
    print(f"📚 {expert.name} v{expert.version} 已启动")
    print(f"专业领域: {', '.join(expert.expertise)}")
    
    # 2. 成本估算
    cost = expert.estimate_analysis_cost("PCSK9")
    print(f"\n💰 分析成本估算:")
    print(f"  预估Token: {cost['estimated_tokens']}")
    print(f"  预估成本: ${cost['estimated_cost_usd']:.4f}")
    print(f"  预估时间: {cost['estimated_time_seconds']}秒")
    print(f"  最大文献数: {cost['max_literature']}")
    
    # 3. 执行分析
    print(f"\n🔬 开始分析 PCSK9...")
    result = await expert.analyze("PCSK9")
    
    # 4. 显示结果
    print(f"\n📊 分析结果:")
    print(f"  基因: {result.gene_target}")
    print(f"  文献数量: {result.total_literature}")
    print(f"  文本块数: {result.total_chunks}")
    print(f"  置信度: {result.confidence_score:.2f}")
    print(f"  分析方法: {result.analysis_method}")
    
    print(f"\n🦠 疾病机制分析:")
    print(result.disease_mechanism[:200] + "...")
    
    print(f"\n💊 治疗策略分析:")
    print(result.treatment_strategy[:200] + "...")
    
    print(f"\n🎯 靶点分析:")
    print(result.target_analysis[:200] + "...")
    
    # 5. Token使用统计
    if result.token_usage:
        print(f"\n💾 Token使用统计:")
        print(f"  总计: {result.token_usage['total_tokens']}")
        print(f"  输入: {result.token_usage['input_tokens']}")
        print(f"  输出: {result.token_usage['output_tokens']}")
    
    # 6. 引用文献
    print(f"\n📚 引用文献 (前5篇):")
    for ref in result.references[:5]:
        print(f"  • {ref['Title']} (PMID: {ref['PMID']})")
    
    print(f"\n✅ 分析完成!")

if __name__ == "__main__":
    asyncio.run(example_usage())
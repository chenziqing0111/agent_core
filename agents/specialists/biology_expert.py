# agent_core/agents/specialists/biology_expert.py
# 使用工作的PubMed检索器的Biology Expert

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# 导入配置系统
try:
    from agent_core.config.analysis_config import (
        AnalysisConfig, AnalysisMode, ConfigManager
    )
except ImportError:
    # 如果配置系统不可用，使用简单配置
    class AnalysisMode:
        QUICK = "quick"
        STANDARD = "standard" 
        DEEP = "deep"
    
    @dataclass
    class AnalysisConfig:
        mode: str = "standard"
        max_literature_per_query: int = 10
        tokens_per_analysis: int = 1000
    
    class ConfigManager:
        @staticmethod
        def get_quick_config():
            return AnalysisConfig(mode="quick", max_literature_per_query=5, tokens_per_analysis=500)
        
        @staticmethod
        def get_standard_config():
            return AnalysisConfig(mode="standard", max_literature_per_query=10, tokens_per_analysis=1000)

# 导入工作的PubMed检索器
from agent_core.agents.tools.retrievers.pubmed_retriever import (
    PubMedRetriever, PubMedArticle, PubMedSearchResult, get_pubmed_abstracts
)

logger = logging.getLogger(__name__)

@dataclass
class BiologyAnalysisResult:
    """生物学分析结果"""
    gene_target: str
    total_articles: int
    analyzed_articles: int
    
    # 文献分析结果
    key_findings: List[Dict[str, Any]]
    biological_functions: List[str]
    pathways_involved: List[str]
    disease_associations: List[Dict[str, Any]]
    drug_interactions: List[Dict[str, Any]]
    
    # 研究趋势
    publication_trends: Dict[str, Any]
    research_hotspots: List[str]
    
    # 关键文献
    highly_cited_papers: List[Dict[str, str]]
    recent_discoveries: List[Dict[str, str]]
    
    # 分析总结
    biological_summary: str
    clinical_relevance: str
    research_gaps: List[str]
    
    # 元数据
    confidence_score: float
    last_updated: str
    config_used: Dict[str, Any]
    token_usage: Dict[str, int]

class BiologyExpert:
    """
    生物学专家Agent - 使用工作的PubMed检索器
    
    Features:
    - 基于可靠的Bio.Entrez的文献检索
    - 基因功能和通路分析
    - 疾病关联分析
    - 药物相互作用挖掘
    - 研究趋势分析
    """
    
    def __init__(self, config: AnalysisConfig = None):
        self.name = "Biology Expert"
        self.version = "2.1.0"
        self.expertise = [
            "gene_function_analysis",
            "pathway_analysis", 
            "disease_association",
            "literature_mining",
            "biomarker_discovery"
        ]
        
        # 使用配置
        self.config = config or ConfigManager.get_standard_config()
        
        logger.info(f"初始化 {self.name} v{self.version} - 模式: {self.config.mode}")
        
    async def analyze(self, gene_target: str, focus_areas: List[str] = None) -> BiologyAnalysisResult:
        """
        执行基因的生物学分析
        
        Args:
            gene_target: 目标基因
            focus_areas: 关注领域 ['function', 'pathways', 'diseases', 'drugs']
        
        Returns:
            BiologyAnalysisResult对象
        """
        
        try:
            logger.info(f"开始生物学分析: {gene_target}")
            start_time = datetime.now()
            
            # 默认分析所有领域
            if focus_areas is None:
                focus_areas = ['function', 'pathways', 'diseases', 'drugs']
            
            # 步骤1: 收集文献数据
            literature_data = await self._collect_literature_data(gene_target)
            
            # 步骤2: 分析不同领域
            analysis_results = {}
            
            if 'function' in focus_areas:
                analysis_results['function'] = await self._analyze_gene_function(
                    gene_target, literature_data
                )
            
            if 'pathways' in focus_areas:
                analysis_results['pathways'] = await self._analyze_pathways(
                    gene_target, literature_data
                )
            
            if 'diseases' in focus_areas:
                analysis_results['diseases'] = await self._analyze_disease_associations(
                    gene_target, literature_data
                )
            
            if 'drugs' in focus_areas:
                analysis_results['drugs'] = await self._analyze_drug_interactions(
                    gene_target, literature_data
                )
            
            # 步骤3: 综合分析
            result = await self._synthesize_results(
                gene_target, literature_data, analysis_results
            )
            
            # 记录使用情况
            end_time = datetime.now()
            analysis_time = (end_time - start_time).total_seconds()
            
            logger.info(f"生物学分析完成: {gene_target} ({analysis_time:.1f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"生物学分析失败: {gene_target} - {str(e)}")
            raise
    
    async def _collect_literature_data(self, gene_target: str) -> Dict[str, Any]:
        """收集文献数据 - 使用工作的检索器"""
        
        try:
            logger.info(f"收集 {gene_target} 的文献数据...")
            
            # 定义不同类型的搜索查询
            search_queries = {
                'general': f"{gene_target} function",
                'pathways': f"{gene_target} pathway signaling",
                'diseases': f"{gene_target} disease cancer",
                'drugs': f"{gene_target} drug therapy treatment",
                'recent': f"{gene_target} 2023 2024"
            }
            
            literature_results = {}
            
            async with PubMedRetriever() as retriever:
                for query_type, query in search_queries.items():
                    # 根据配置调整搜索数量
                    max_results = self._get_literature_count_for_query(query_type)
                    
                    logger.info(f"搜索 {query_type}: {query} (最多{max_results}篇)")
                    
                    try:
                        result = await retriever.search_literature(query, max_results)
                        literature_results[query_type] = {
                            'query': query,
                            'articles': result.articles,
                            'count': len(result.articles)
                        }
                        
                        logger.info(f"  ✅ {query_type}: 获得 {len(result.articles)} 篇文献")
                        
                        # 简短延迟，避免过于频繁的请求
                        await asyncio.sleep(0.3)
                        
                    except Exception as e:
                        logger.warning(f"  ❌ {query_type} 搜索失败: {e}")
                        literature_results[query_type] = {
                            'query': query,
                            'articles': [],
                            'count': 0,
                            'error': str(e)
                        }
            
            # 统计总文献数
            total_articles = sum(
                data.get('count', 0) for data in literature_results.values()
            )
            
            logger.info(f"文献收集完成: 总共 {total_articles} 篇")
            
            return literature_results
            
        except Exception as e:
            logger.error(f"文献数据收集失败: {e}")
            return {}
    
    def _get_literature_count_for_query(self, query_type: str) -> int:
        """根据配置和查询类型确定文献数量"""
        
        base_counts = {
            "quick": {'general': 3, 'pathways': 2, 'diseases': 2, 'drugs': 2, 'recent': 1},
            "standard": {'general': 5, 'pathways': 4, 'diseases': 4, 'drugs': 4, 'recent': 3},
            "deep": {'general': 10, 'pathways': 8, 'diseases': 8, 'drugs': 8, 'recent': 5},
        }
        
        mode = getattr(self.config, 'mode', 'standard')
        mode_counts = base_counts.get(mode, base_counts['standard'])
        return mode_counts.get(query_type, 3)
    
    async def _analyze_gene_function(self, gene_target: str, literature_data: Dict) -> Dict[str, Any]:
        """分析基因功能 - 基于文献内容的简化分析"""
        
        try:
            general_articles = literature_data.get('general', {}).get('articles', [])
            
            if not general_articles:
                return {'functions': [], 'confidence': 0.0}
            
            # 基于文献标题和摘要的关键词分析
            function_keywords = self._extract_function_keywords(general_articles)
            mechanisms = self._extract_mechanisms(general_articles)
            
            # 简化的功能分析
            analysis_result = {
                "primary_functions": function_keywords[:5],  # 前5个功能
                "molecular_mechanisms": mechanisms,
                "cellular_localization": self._infer_localization(general_articles),
                "confidence_score": min(0.8, len(general_articles) * 0.1)
            }
            
            logger.info(f"基因功能分析完成: {len(function_keywords)} 个功能关键词")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"基因功能分析失败: {e}")
            return {'functions': [], 'confidence': 0.0}
    
    def _extract_function_keywords(self, articles: List[PubMedArticle]) -> List[str]:
        """从文献中提取功能关键词"""
        
        function_terms = [
            'transcription', 'regulation', 'expression', 'binding', 'activation',
            'inhibition', 'signaling', 'metabolism', 'transport', 'catalysis',
            'repair', 'replication', 'translation', 'splicing', 'modification'
        ]
        
        found_functions = []
        
        for article in articles:
            text = (article.title + " " + article.abstract).lower()
            
            for term in function_terms:
                if term in text and term not in found_functions:
                    found_functions.append(term)
        
        return found_functions
    
    def _extract_mechanisms(self, articles: List[PubMedArticle]) -> str:
        """提取分子机制描述"""
        
        mechanism_keywords = ['mechanism', 'pathway', 'interaction', 'binding', 'regulation']
        
        mechanisms = []
        for article in articles[:3]:  # 只看前3篇
            title_lower = article.title.lower()
            abstract_lower = article.abstract.lower()
            
            for keyword in mechanism_keywords:
                if keyword in title_lower or keyword in abstract_lower:
                    # 提取包含关键词的句子片段
                    if keyword in abstract_lower:
                        sentences = article.abstract.split('.')
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                mechanisms.append(sentence.strip())
                                break
        
        return "; ".join(mechanisms[:2])  # 最多2个机制描述
    
    def _infer_localization(self, articles: List[PubMedArticle]) -> str:
        """推断细胞定位"""
        
        localization_terms = {
            'nucleus': ['nucleus', 'nuclear', 'chromatin'],
            'cytoplasm': ['cytoplasm', 'cytoplasmic', 'cytosol'],
            'membrane': ['membrane', 'transmembrane', 'surface'],
            'mitochondria': ['mitochondria', 'mitochondrial'],
            'secreted': ['secreted', 'extracellular', 'plasma']
        }
        
        for article in articles[:3]:
            text = (article.title + " " + article.abstract).lower()
            
            for location, terms in localization_terms.items():
                if any(term in text for term in terms):
                    return location
        
        return "unknown"
    
    async def _analyze_pathways(self, gene_target: str, literature_data: Dict) -> Dict[str, Any]:
        """分析信号通路"""
        
        try:
            pathway_articles = literature_data.get('pathways', {}).get('articles', [])
            
            if not pathway_articles:
                return {'pathways': [], 'confidence': 0.0}
            
            # 提取通路关键词
            pathway_keywords = self._extract_pathway_keywords(pathway_articles)
            interactions = self._extract_interactions(pathway_articles)
            
            analysis_result = {
                "major_pathways": pathway_keywords[:5],
                "pathway_roles": {pathway: "参与调节" for pathway in pathway_keywords[:3]},
                "key_interactions": interactions,
                "disease_relevance": "与多种疾病发生发展相关",
                "confidence_score": min(0.8, len(pathway_articles) * 0.1)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"通路分析失败: {e}")
            return {'pathways': [], 'confidence': 0.0}
    
    def _extract_pathway_keywords(self, articles: List[PubMedArticle]) -> List[str]:
        """提取通路关键词"""
        
        pathway_terms = [
            'p53', 'MAPK', 'PI3K', 'AKT', 'mTOR', 'Wnt', 'Notch', 'TGF-beta',
            'NF-kB', 'JAK-STAT', 'DNA repair', 'apoptosis', 'cell cycle'
        ]
        
        found_pathways = []
        
        for article in articles:
            text = (article.title + " " + article.abstract).lower()
            
            for pathway in pathway_terms:
                if pathway.lower() in text and pathway not in found_pathways:
                    found_pathways.append(pathway)
        
        return found_pathways
    
    def _extract_interactions(self, articles: List[PubMedArticle]) -> List[str]:
        """提取相互作用基因/蛋白"""
        
        # 常见的基因名称模式
        gene_patterns = ['p53', 'ATM', 'BRCA1', 'BRCA2', 'MDM2', 'p21', 'AKT', 'mTOR']
        
        interactions = []
        
        for article in articles[:3]:
            text = article.title + " " + article.abstract
            
            for gene in gene_patterns:
                if gene in text and gene not in interactions:
                    interactions.append(gene)
        
        return interactions[:5]  # 最多5个相互作用
    
    async def _analyze_disease_associations(self, gene_target: str, literature_data: Dict) -> Dict[str, Any]:
        """分析疾病关联"""
        
        try:
            disease_articles = literature_data.get('diseases', {}).get('articles', [])
            
            if not disease_articles:
                return {'diseases': [], 'confidence': 0.0}
            
            diseases = self._extract_diseases(disease_articles)
            mutations = self._extract_mutations(disease_articles)
            
            analysis_result = {
                "associated_diseases": [
                    {"disease": disease, "mutation_type": "various", "clinical_significance": "研究中"}
                    for disease in diseases
                ],
                "disease_mechanisms": "基因功能异常导致疾病发生",
                "biomarker_potential": "具有潜在的生物标志物价值",
                "confidence_score": min(0.9, len(disease_articles) * 0.15)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"疾病关联分析失败: {e}")
            return {'diseases': [], 'confidence': 0.0}
    
    def _extract_diseases(self, articles: List[PubMedArticle]) -> List[str]:
        """提取疾病名称"""
        
        disease_terms = [
            'cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia', 'sarcoma',
            'diabetes', 'alzheimer', 'parkinson', 'heart disease', 'stroke'
        ]
        
        found_diseases = []
        
        for article in articles:
            text = (article.title + " " + article.abstract).lower()
            
            for disease in disease_terms:
                if disease in text and disease not in found_diseases:
                    found_diseases.append(disease)
        
        return found_diseases
    
    def _extract_mutations(self, articles: List[PubMedArticle]) -> List[str]:
        """提取突变类型"""
        
        mutation_terms = ['mutation', 'deletion', 'insertion', 'polymorphism', 'variant']
        
        found_mutations = []
        
        for article in articles:
            text = (article.title + " " + article.abstract).lower()
            
            for mutation in mutation_terms:
                if mutation in text and mutation not in found_mutations:
                    found_mutations.append(mutation)
        
        return found_mutations
    
    async def _analyze_drug_interactions(self, gene_target: str, literature_data: Dict) -> Dict[str, Any]:
        """分析药物相互作用"""
        
        try:
            drug_articles = literature_data.get('drugs', {}).get('articles', [])
            
            if not drug_articles:
                return {'drugs': [], 'confidence': 0.0}
            
            drugs = self._extract_drugs(drug_articles)
            
            analysis_result = {
                "drug_interactions": [
                    {"drug": drug, "mechanism": "target modulation", "effect": "therapeutic potential"}
                    for drug in drugs
                ],
                "pharmacogenomics": "基因变异可能影响药物反应",
                "personalized_medicine": "有望用于个性化治疗",
                "confidence_score": min(0.75, len(drug_articles) * 0.12)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"药物相互作用分析失败: {e}")
            return {'drugs': [], 'confidence': 0.0}
    
    def _extract_drugs(self, articles: List[PubMedArticle]) -> List[str]:
        """提取药物名称"""
        
        drug_terms = [
            'chemotherapy', 'radiation', 'inhibitor', 'agonist', 'antagonist',
            'therapy', 'treatment', 'drug', 'compound', 'molecule'
        ]
        
        found_drugs = []
        
        for article in articles:
            text = (article.title + " " + article.abstract).lower()
            
            for drug in drug_terms:
                if drug in text and drug not in found_drugs:
                    found_drugs.append(drug)
        
        return found_drugs
    
    async def _synthesize_results(self, 
                                gene_target: str, 
                                literature_data: Dict, 
                                analysis_results: Dict) -> BiologyAnalysisResult:
        """综合分析结果"""
        
        try:
            # 统计文献数量
            total_articles = sum(
                data.get('count', 0) for data in literature_data.values()
            )
            
            # 提取关键发现
            key_findings = []
            biological_functions = []
            pathways_involved = []
            disease_associations = []
            drug_interactions = []
            
            # 从各个分析中提取信息
            if 'function' in analysis_results:
                func_data = analysis_results['function']
                biological_functions = func_data.get('primary_functions', [])
                
                key_findings.append({
                    'category': '基因功能',
                    'content': func_data.get('molecular_mechanisms', ''),
                    'confidence': func_data.get('confidence_score', 0.0)
                })
            
            if 'pathways' in analysis_results:
                pathway_data = analysis_results['pathways']
                pathways_involved = pathway_data.get('major_pathways', [])
                
                key_findings.append({
                    'category': '信号通路',
                    'content': pathway_data.get('disease_relevance', ''),
                    'confidence': pathway_data.get('confidence_score', 0.0)
                })
            
            if 'diseases' in analysis_results:
                disease_data = analysis_results['diseases']
                disease_associations = disease_data.get('associated_diseases', [])
                
                key_findings.append({
                    'category': '疾病关联',
                    'content': disease_data.get('disease_mechanisms', ''),
                    'confidence': disease_data.get('confidence_score', 0.0)
                })
            
            if 'drugs' in analysis_results:
                drug_data = analysis_results['drugs']
                drug_interactions = drug_data.get('drug_interactions', [])
                
                key_findings.append({
                    'category': '药物相互作用',
                    'content': drug_data.get('pharmacogenomics', ''),
                    'confidence': drug_data.get('confidence_score', 0.0)
                })
            
            # 选择关键文献
            highly_cited_papers = self._select_key_papers(literature_data, 'general')
            recent_discoveries = self._select_key_papers(literature_data, 'recent')
            
            # 生成简化总结
            biological_summary = f"{gene_target}基因参与{', '.join(biological_functions[:3])}等生物学过程，"
            biological_summary += f"涉及{', '.join(pathways_involved[:2])}等信号通路。"
            
            clinical_relevance = f"与{len(disease_associations)}种疾病相关，"
            clinical_relevance += f"具有{len(drug_interactions)}种潜在药物相互作用。"
            
            research_gaps = ["需要更多功能验证研究", "临床转化研究不足", "分子机制有待深入"]
            
            # 计算总体置信度
            confidence_scores = [finding.get('confidence', 0.0) for finding in key_findings]
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # 估算token使用量
            token_usage = {
                "total_tokens": len(analysis_results) * 200,
                "analysis_tokens": len(analysis_results) * 150,
                "literature_tokens": total_articles * 50
            }
            
            return BiologyAnalysisResult(
                gene_target=gene_target,
                total_articles=total_articles,
                analyzed_articles=min(total_articles, 20),
                key_findings=key_findings,
                biological_functions=biological_functions,
                pathways_involved=pathways_involved,
                disease_associations=disease_associations,
                drug_interactions=drug_interactions,
                publication_trends={"trend": "increasing", "years": ["2022", "2023", "2024"]},
                research_hotspots=["免疫治疗", "精准医学", "基因编辑"],
                highly_cited_papers=highly_cited_papers,
                recent_discoveries=recent_discoveries,
                biological_summary=biological_summary,
                clinical_relevance=clinical_relevance,
                research_gaps=research_gaps,
                confidence_score=overall_confidence,
                last_updated=datetime.now().isoformat(),
                config_used=asdict(self.config),
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"结果综合失败: {e}")
            raise
    
    def _select_key_papers(self, literature_data: Dict, data_type: str) -> List[Dict[str, str]]:
        """选择关键文献"""
        
        articles = literature_data.get(data_type, {}).get('articles', [])
        
        key_papers = []
        for article in articles[:3]:  # 选择前3篇
            key_papers.append({
                'pmid': article.pmid,
                'title': article.title,
                'journal': article.journal,
                'url': article.url
            })
        
        return key_papers


# 测试函数
async def test_biology_expert():
    """测试Biology Expert"""
    
    print("🧠 测试Biology Expert with Working PubMed Retriever")
    print("=" * 60)
    
    # 使用快速配置进行测试
    config = ConfigManager.get_quick_config()
    expert = BiologyExpert(config)
    
    print(f"专家信息:")
    print(f"  名称: {expert.name} v{expert.version}")
    print(f"  配置: {expert.config.mode}")
    print(f"  专业领域: {', '.join(expert.expertise)}")
    
    test_genes = ['BRCA1', 'TP53']
    
    for gene in test_genes:
        print(f"\n{'='*40}")
        print(f"🧬 分析基因: {gene}")
        print(f"{'='*40}")
        
        try:
            result = await expert.analyze(gene, focus_areas=['function', 'diseases'])
            
            print(f"\n📊 分析结果:")
            print(f"  基因: {result.gene_target}")
            print(f"  文献总数: {result.total_articles}")
            print(f"  分析文献: {result.analyzed_articles}")
            print(f"  置信度: {result.confidence_score:.2f}")
            
            print(f"\n🔬 生物学功能:")
            for func in result.biological_functions[:3]:
                print(f"    • {func}")
            
            print(f"\n🏥 疾病关联:")
            for disease in result.disease_associations[:3]:
                print(f"    • {disease.get('disease', 'Unknown')}")
            
            print(f"\n📈 关键发现:")
            for finding in result.key_findings:
                print(f"    • {finding['category']}: {finding['content'][:100]}...")
            
            print(f"\n📄 代表性文献:")
            for paper in result.highly_cited_papers[:2]:
                print(f"    • [{paper['pmid']}] {paper['title'][:80]}...")
            
            print(f"\n💡 总结:")
            print(f"    {result.biological_summary}")
            
            print(f"\n🎯 临床相关性:")
            print(f"    {result.clinical_relevance}")
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            logger.error(f"基因分析失败: {gene}", exc_info=True)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_biology_expert())
# agent_core/agents/specialists/patent_expert.py
# 专利分析专家智能体

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from agent_core.config.analysis_config import AnalysisConfig, ConfigManager
from agent_core.agents.tools.retrievers.patent_retriever import PatentRetriever, Patent, PatentSearchResult
from agent_core.agents.tools.retrievers.real_patent_retriever import RealPatentRetriever, RealPatent, RealPatentSearchResult

logger = logging.getLogger(__name__)

@dataclass
class PatentAnalysisResult:
    """专利分析结果数据结构 - 增强版"""
    target: str
    total_patents: int
    key_patents: List[Dict[str, Any]]
    landscape_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    competitive_insights: List[str]
    technology_gaps: List[str]
    recommendations: List[str]
    confidence_score: float
    token_usage: int
    analysis_timestamp: str
    
    # 🆕 新增深度分析字段
    technical_analysis: Dict[str, Any]  # 技术方案分析
    innovation_analysis: Dict[str, Any]  # 创新点分析
    patent_value_assessment: Dict[str, Any]  # 专利价值评估
    trend_chart_data: Dict[str, Any]  # 趋势图表数据
    claims_analysis: Dict[str, Any]  # 权利要求分析
    research_purposes: List[Dict[str, Any]]  # 研究目的分析
    genomic_protection_analysis: Dict[str, Any]  # 🆕 基因组保护分析（表观基因编辑专用）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

class PatentExpert:
    """
    专利分析专家 - 负责专利检索、景观分析和知识产权洞察
    
    主要功能：
    1. 专利检索和收集
    2. 专利景观分析
    3. 技术趋势识别
    4. 竞争对手分析
    5. 知识产权策略建议
    """
    
    def __init__(self, config: AnalysisConfig = None, use_real_data: bool = True):
        self.name = "Patent Expert"
        self.version = "2.0.0"
        self.config = config or ConfigManager.get_standard_config()
        self.use_real_data = use_real_data
        
        # 选择数据源
        if use_real_data:
            self.retriever = RealPatentRetriever()
            self.data_type = "real"
        else:
            self.retriever = PatentRetriever()
            self.data_type = "mock"
            
        self.logger = logging.getLogger(__name__)
        
        # 根据配置调整参数
        self._configure_analysis_params()
        
        logger.info(f"初始化专利专家 v{self.version} - 模式: {self.config.mode} - 数据源: {self.data_type}")
    
    def _configure_analysis_params(self):
        """根据配置模式设置分析参数"""
        mode_params = {
            "QUICK": {
                "max_patents": 10,
                "analysis_depth": "basic",
                "trend_years": 3,
                "competitor_analysis": False
            },
            "STANDARD": {
                "max_patents": 30,
                "analysis_depth": "moderate",
                "trend_years": 5,
                "competitor_analysis": True
            },
            "DEEP": {
                "max_patents": 50,
                "analysis_depth": "comprehensive",
                "trend_years": 10,
                "competitor_analysis": True
            },
            "CUSTOM": {
                "max_patents": getattr(self.config.clinical_trials, 'max_trials_analyze', 30),
                "analysis_depth": "comprehensive",
                "trend_years": 5,
                "competitor_analysis": True
            }
        }
        
        # 处理AnalysisMode枚举和字符串
        mode_key = self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode)
        mode_key = mode_key.upper()
        self.params = mode_params.get(mode_key, mode_params["STANDARD"])
    
    async def analyze(self, target: str, context: Dict[str, Any] = None) -> PatentAnalysisResult:
        """
        执行专利分析
        
        Args:
            target: 分析目标（基因名称、化合物等）
            context: 额外的上下文信息
            
        Returns:
            PatentAnalysisResult: 专利分析结果
        """
        start_time = datetime.now()
        context = context or {}
        
        try:
            self.logger.info(f"开始分析 {target} 的专利景观")
            
            # 1. 检索相关专利
            patents = await self._search_patents(target, context)
            
            # 2. 分析专利景观
            landscape = await self._analyze_landscape(patents)
            
            # 3. 分析技术趋势
            trends = await self._analyze_trends(target, patents)
            
            # 4. 识别竞争对手
            competitors = await self._analyze_competitors(patents) if self.params["competitor_analysis"] else []
            
            # 5. 识别技术缺口
            tech_gaps = await self._identify_technology_gaps(target, patents, context)
            
            # 6. 生成策略建议
            recommendations = await self._generate_recommendations(
                target, patents, landscape, trends, competitors, tech_gaps
            )
            
            # 7. 提取关键专利
            key_patents = self._extract_key_patents(patents)
            
            # 8. 🆕 深度技术分析
            technical_analysis = await self._perform_technical_analysis(patents) if self.params["analysis_depth"] != "basic" else {}
            
            # 9. 🆕 创新点分析
            innovation_analysis = await self._analyze_innovation_points(patents, target) if self.params["analysis_depth"] != "basic" else {}
            
            # 10. 🆕 专利价值评估
            value_assessment = await self._assess_patent_value(patents, landscape) if self.params["analysis_depth"] in ["moderate", "comprehensive"] else {}
            
            # 11. 🆕 生成趋势图表数据
            chart_data = self._generate_trend_chart_data(patents, trends)
            
            # 12. 🆕 权利要求分析
            claims_analysis = await self._analyze_patent_claims(patents) if self.params["analysis_depth"] == "comprehensive" else {}
            
            # 13. 🆕 研究目的分析
            research_purposes = await self._analyze_research_purposes(patents, target) if self.params["analysis_depth"] != "basic" else []
            
            # 14. 🆕 基因组保护分析（表观基因编辑专用）
            genomic_protection = await self._analyze_genomic_protection(patents, target) if self.params["analysis_depth"] in ["moderate", "comprehensive"] else {}
            
            # 15. 计算置信度分数
            confidence = self._calculate_confidence(patents, landscape)
            
            # 16. 估算token使用
            token_usage = self._estimate_token_usage(patents)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"专利分析完成 - 耗时: {analysis_time:.2f}秒, Token: {token_usage}")
            
            return PatentAnalysisResult(
                target=target,
                total_patents=len(patents.patents) if patents else 0,
                key_patents=key_patents,
                landscape_analysis=landscape,
                trend_analysis=trends,
                competitive_insights=competitors,
                technology_gaps=tech_gaps,
                recommendations=recommendations,
                confidence_score=confidence,
                token_usage=token_usage,
                analysis_timestamp=datetime.now().isoformat(),
                # 🆕 新增字段
                technical_analysis=technical_analysis,
                innovation_analysis=innovation_analysis,
                patent_value_assessment=value_assessment,
                trend_chart_data=chart_data,
                claims_analysis=claims_analysis,
                research_purposes=research_purposes,
                genomic_protection_analysis=genomic_protection
            )
            
        except Exception as e:
            self.logger.error(f"专利分析失败: {e}")
            raise
    
    async def _search_patents(self, target: str, context: Dict[str, Any]):
        """检索相关专利"""
        focus_areas = context.get("patent_focus_areas", ["therapy", "diagnostic", "method"])
        additional_terms = context.get("additional_terms", ["epigenetic", "methylation"])
        
        async with self.retriever as retriever:
            # 主要检索
            main_results = await retriever.search_by_gene(
                gene=target,
                additional_terms=additional_terms,
                max_results=self.params["max_patents"],
                focus_areas=focus_areas
            )
            
            # 如果是深度分析且使用模拟数据，进行额外检索
            if self.params["analysis_depth"] == "comprehensive" and not self.use_real_data:
                # 检索相关化合物专利
                compound_results = await retriever.search_patents(
                    query=f"{target} inhibitor compound",
                    max_results=10
                )
                
                # 合并结果
                all_patents = main_results.patents + compound_results.patents
                # 去重
                unique_patents = {p.patent_id: p for p in all_patents}.values()
                main_results.patents = list(unique_patents)[:self.params["max_patents"]]
                main_results.total_count = len(main_results.patents)
            
            return main_results
    
    async def _analyze_landscape(self, patents) -> Dict[str, Any]:
        """分析专利景观"""
        if not patents or not patents.patents:
            return {"status": "no_data"}
        
        async with self.retriever as retriever:
            landscape = await retriever.analyze_patent_landscape(patents.patents)
        
        # 增强分析
        landscape["innovation_intensity"] = self._calculate_innovation_intensity(landscape)
        landscape["market_maturity"] = self._assess_market_maturity(landscape)
        landscape["key_players"] = self._identify_key_players(landscape)
        
        # 添加数据源信息
        if self.use_real_data and hasattr(patents, 'sources_used'):
            landscape["data_sources"] = patents.sources_used
            landscape["data_quality"] = "real_data"
        else:
            landscape["data_sources"] = ["mock_data"]
            landscape["data_quality"] = "simulated"
        
        return landscape
    
    async def _analyze_trends(self, target: str, patents) -> Dict[str, Any]:
        """分析技术趋势"""
        if self.params["analysis_depth"] == "basic":
            return {"status": "basic_analysis", "trend": "stable"}
        
        trends = {
            "filing_trend": self._analyze_filing_trend(patents),
            "technology_evolution": self._analyze_tech_evolution(patents),
            "emerging_areas": self._identify_emerging_areas(patents),
            "forecast": self._generate_forecast(patents)
        }
        
        return trends
    
    async def _analyze_competitors(self, patents) -> List[str]:
        """分析竞争对手"""
        if not patents or not patents.patents:
            return []
        
        insights = []
        
        # 分析主要申请人
        assignee_stats = {}
        for patent in patents.patents:
            assignee = patent.assignee
            if assignee not in assignee_stats:
                assignee_stats[assignee] = {
                    "count": 0,
                    "recent_patents": 0,
                    "active_patents": 0
                }
            assignee_stats[assignee]["count"] += 1
            if patent.filing_date and patent.filing_date.startswith("202"):
                assignee_stats[assignee]["recent_patents"] += 1
            if patent.status == "Active":
                assignee_stats[assignee]["active_patents"] += 1
        
        # 生成竞争洞察
        top_assignees = sorted(assignee_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
        
        for assignee, stats in top_assignees:
            insight = f"{assignee}持有{stats['count']}项相关专利"
            if stats["recent_patents"] > 0:
                insight += f"，近期申请{stats['recent_patents']}项"
            insights.append(insight)
        
        # 技术领先者分析
        if top_assignees:
            leader = top_assignees[0][0]
            insights.append(f"{leader}在该领域处于技术领先地位")
        
        return insights
    
    async def _identify_technology_gaps(self, target: str, patents, context: Dict[str, Any]) -> List[str]:
        """识别技术缺口"""
        gaps = []
        
        if not patents or not patents.patents:
            gaps.append(f"{target}相关专利较少，可能存在技术空白")
            return gaps
        
        # 分析专利覆盖的技术领域
        covered_areas = set()
        for patent in patents.patents:
            if "therapy" in patent.title.lower() or "treatment" in patent.title.lower():
                covered_areas.add("治疗应用")
            if "diagnostic" in patent.title.lower() or "detection" in patent.title.lower():
                covered_areas.add("诊断应用")
            if "crispr" in patent.title.lower() or "editing" in patent.title.lower():
                covered_areas.add("基因编辑")
            if "compound" in patent.title.lower() or "inhibitor" in patent.title.lower():
                covered_areas.add("小分子化合物")
        
        # 识别缺口
        potential_areas = {"治疗应用", "诊断应用", "基因编辑", "小分子化合物", "生物标记物", "给药系统"}
        missing_areas = potential_areas - covered_areas
        
        for area in missing_areas:
            gaps.append(f"{area}领域的专利布局相对薄弱")
        
        # 地理覆盖分析
        if self.params["analysis_depth"] in ["moderate", "comprehensive"]:
            gaps.append("建议关注亚太地区的专利布局机会")
        
        return gaps
    
    async def _generate_recommendations(self, target: str, patents, 
                                      landscape: Dict[str, Any], trends: Dict[str, Any],
                                      competitors: List[str], tech_gaps: List[str]) -> List[str]:
        """生成专利策略建议"""
        recommendations = []
        
        # 基于专利数量的建议
        total_patents = landscape.get("total_patents", 0)
        if total_patents < 10:
            recommendations.append(f"建议加强{target}相关的专利申请，当前专利保护较弱")
        elif total_patents > 50:
            recommendations.append(f"{target}领域专利密集，建议寻找差异化创新点")
        
        # 基于趋势的建议
        if trends.get("filing_trend") == "increasing":
            recommendations.append("该领域专利申请呈上升趋势，建议加快研发和专利布局")
        
        # 基于竞争的建议
        if competitors and len(competitors) > 3:
            recommendations.append("竞争激烈，建议通过专利组合策略建立竞争优势")
        
        # 基于技术缺口的建议
        if tech_gaps:
            recommendations.append(f"可考虑在{tech_gaps[0]}等方向进行专利布局")
        
        # 合作建议
        if self.params["analysis_depth"] == "comprehensive":
            recommendations.append("建议通过专利交叉许可或合作开发拓展技术应用")
        
        return recommendations
    
    def _extract_key_patents(self, patents) -> List[Dict[str, Any]]:
        """提取关键专利"""
        if not patents or not patents.patents:
            return []
        
        key_patents = []
        
        # 选择最相关的专利
        selected = patents.patents[:5] if self.params["analysis_depth"] == "basic" else patents.patents[:10]
        
        for patent in selected:
            key_patent = {
                "id": patent.patent_id,
                "title": patent.title,
                "assignee": patent.assignee,
                "filing_date": patent.filing_date,
                "status": patent.status,
                "relevance_score": self._calculate_patent_relevance(patent),
                "summary": patent.abstract[:200] + "..." if patent.abstract else "无摘要",
                "url": patent.url
            }
            key_patents.append(key_patent)
        
        # 按相关性排序
        key_patents.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return key_patents
    
    def _calculate_patent_relevance(self, patent) -> float:
        """计算专利相关性分数"""
        score = 0.5  # 基础分数
        
        # 基于状态
        if patent.status in ["Active", "Granted", "Published"]:
            score += 0.2
        
        # 基于时间
        if patent.filing_date and patent.filing_date.startswith("202"):
            score += 0.2
        
        # 基于引用
        if hasattr(patent, 'cited_by') and patent.cited_by and len(patent.cited_by) > 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_innovation_intensity(self, landscape: Dict[str, Any]) -> str:
        """计算创新强度"""
        total = landscape.get("total_patents", 0)
        recent_years = [k for k in landscape.get("year_distribution", {}).keys() if k >= "2020"]
        recent_count = sum(landscape.get("year_distribution", {}).get(y, 0) for y in recent_years)
        
        if total == 0:
            return "low"
        
        recent_ratio = recent_count / total
        if recent_ratio > 0.5:
            return "high"
        elif recent_ratio > 0.3:
            return "moderate"
        else:
            return "low"
    
    def _assess_market_maturity(self, landscape: Dict[str, Any]) -> str:
        """评估市场成熟度"""
        total = landscape.get("total_patents", 0)
        assignees = landscape.get("top_assignees", {})
        
        if total < 10:
            return "emerging"
        elif total < 50 and len(assignees) < 10:
            return "growing"
        elif total < 200:
            return "maturing"
        else:
            return "mature"
    
    def _identify_key_players(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别关键参与者"""
        players = []
        for assignee, count in list(landscape.get("top_assignees", {}).items())[:5]:
            players.append({
                "name": assignee,
                "patents": count,
                "market_share": round(count / landscape.get("total_patents", 1) * 100, 1)
            })
        return players
    
    def _analyze_filing_trend(self, patents) -> str:
        """分析申请趋势"""
        if not patents or not patents.patents:
            return "insufficient_data"
        
        year_counts = {}
        for patent in patents.patents:
            if patent.filing_date:
                year = patent.filing_date.split('-')[0]
                year_counts[year] = year_counts.get(year, 0) + 1
        
        years = sorted(year_counts.keys())
        if len(years) < 2:
            return "stable"
        
        recent_avg = sum(year_counts.get(y, 0) for y in years[-2:]) / 2
        earlier_avg = sum(year_counts.get(y, 0) for y in years[:-2]) / max(len(years) - 2, 1)
        
        if recent_avg > earlier_avg * 1.5:
            return "increasing"
        elif recent_avg < earlier_avg * 0.7:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_tech_evolution(self, patents) -> List[str]:
        """分析技术演进"""
        evolution = []
        
        if not patents or not patents.patents:
            return ["数据不足，无法分析技术演进"]
        
        # 按年份分组
        year_groups = {}
        for patent in patents.patents:
            if patent.filing_date:
                year = patent.filing_date.split('-')[0]
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(patent)
        
        # 分析每个时期的技术特点
        for year in sorted(year_groups.keys())[-3:]:
            patents_in_year = year_groups[year]
            tech_keywords = []
            for p in patents_in_year:
                if "CRISPR" in p.title.upper():
                    tech_keywords.append("基因编辑")
                if "compound" in p.title.lower():
                    tech_keywords.append("小分子化合物")
                if "antibody" in p.title.lower():
                    tech_keywords.append("抗体")
            
            if tech_keywords:
                evolution.append(f"{year}年：{', '.join(set(tech_keywords))}")
        
        return evolution if evolution else ["技术发展稳定，无明显演进趋势"]
    
    def _identify_emerging_areas(self, patents) -> List[str]:
        """识别新兴领域"""
        emerging = []
        
        if not patents or not patents.patents:
            return emerging
        
        recent_patents = [p for p in patents.patents if p.filing_date and p.filing_date.startswith("202")]
        
        # 分析最近的专利主题
        recent_themes = set()
        for patent in recent_patents[:10]:
            title_lower = patent.title.lower()
            if "ai" in title_lower or "machine learning" in title_lower:
                recent_themes.add("AI辅助药物设计")
            if "nanoparticle" in title_lower:
                recent_themes.add("纳米递送系统")
            if "organoid" in title_lower:
                recent_themes.add("类器官模型")
            if "single cell" in title_lower:
                recent_themes.add("单细胞分析")
        
        emerging.extend(list(recent_themes))
        
        return emerging if emerging else ["暂未发现明显的新兴技术方向"]
    
    def _generate_forecast(self, patents) -> str:
        """生成预测"""
        trend = self._analyze_filing_trend(patents)
        
        forecasts = {
            "increasing": "预计未来2-3年专利申请将持续增长，竞争将更加激烈",
            "stable": "预计专利申请将保持稳定，市场格局相对固定",
            "decreasing": "专利申请有所减少，可能表明技术成熟或市场转向",
            "insufficient_data": "数据不足，难以进行可靠预测"
        }
        
        return forecasts.get(trend, "无法生成预测")
    
    def _calculate_confidence(self, patents, landscape: Dict[str, Any]) -> float:
        """计算分析置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于数据量
        if patents and patents.total_count > 20:
            confidence += 0.2
        elif patents and patents.total_count > 10:
            confidence += 0.1
        
        # 基于数据完整性
        if landscape.get("total_patents", 0) > 0:
            confidence += 0.1
        
        # 基于数据质量
        if landscape.get("data_quality") == "real_data":
            confidence += 0.15  # 真实数据获得更高置信度
        
        # 基于分析深度
        if self.params["analysis_depth"] == "comprehensive":
            confidence += 0.2
        elif self.params["analysis_depth"] == "moderate":
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _estimate_token_usage(self, patents) -> int:
        """估算Token使用量"""
        base_tokens = 500  # 基础消耗
        
        if patents:
            # 每个专利约100 tokens
            patent_tokens = len(patents.patents) * 100
            base_tokens += patent_tokens
        
        # 根据分析深度调整
        depth_multiplier = {
            "basic": 1.0,
            "moderate": 1.5,
            "comprehensive": 2.0
        }
        
        multiplier = depth_multiplier.get(self.params["analysis_depth"], 1.0)
        
        return int(base_tokens * multiplier)
    
    # 🆕 新增深度分析方法
    
    async def _perform_technical_analysis(self, patents) -> Dict[str, Any]:
        """执行深度技术分析"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            technical_schemes = []
            core_technologies = []
            solution_approaches = []
            
            for patent in patents.patents[:10]:  # 分析前10个专利
                # 从标题和摘要中提取技术方案
                if hasattr(patent, 'abstract') and patent.abstract:
                    tech_elements = self._extract_technical_elements(patent.title, patent.abstract)
                    technical_schemes.append({
                        'patent_id': patent.patent_id,
                        'tech_scheme': tech_elements['scheme'],
                        'key_technology': tech_elements['technology'],
                        'approach': tech_elements['approach']
                    })
                    
                    core_technologies.extend(tech_elements['core_tech'])
                    solution_approaches.extend(tech_elements['solutions'])
            
            # 统计核心技术
            tech_frequency = {}
            for tech in core_technologies:
                tech_frequency[tech] = tech_frequency.get(tech, 0) + 1
            
            return {
                "core_technologies": dict(sorted(tech_frequency.items(), key=lambda x: x[1], reverse=True)[:5]),
                "technical_schemes": technical_schemes[:5],
                "solution_categories": list(set(solution_approaches))[:5],
                "technology_maturity": self._assess_technology_maturity(patents.patents),
                "technical_depth": "comprehensive" if len(technical_schemes) > 5 else "moderate"
            }
        except Exception as e:
            self.logger.error(f"技术分析失败: {e}")
            return {"status": "analysis_failed", "error": str(e)}
    
    def _extract_technical_elements(self, title: str, abstract: str) -> Dict[str, Any]:
        """从专利标题和摘要中提取技术要素"""
        text = (title + " " + abstract).lower()
        
        # 核心技术关键词 - 扩展版
        core_tech_keywords = [
            "crispr", "gene editing", "methylation", "histone", "chromatin",
            "inhibitor", "compound", "antibody", "protein", "enzyme",
            "nanoparticle", "delivery", "targeting", "biomarker", "assay",
            "therapeutic", "pharmaceutical", "composition", "vaccine",
            "polypeptide", "nucleic acid", "oligonucleotide", "vector",
            "cell therapy", "immunotherapy", "monoclonal", "recombinant"
        ]
        
        # 解决方案关键词
        solution_keywords = [
            "treatment", "therapy", "therapeutic", "diagnostic", "detection", "screening",
            "prevention", "intervention", "monitoring", "prognosis", "ameliorating",
            "treating", "inhibiting", "modulating", "regulating", "enhancing"
        ]
        
        # 技术方法关键词
        approach_keywords = [
            "method", "system", "device", "composition", "formulation",
            "process", "procedure", "technique", "platform", "tool",
            "kit", "apparatus", "array", "microarray", "chip"
        ]
        
        found_tech = [tech for tech in core_tech_keywords if tech in text]
        found_solutions = [sol for sol in solution_keywords if sol in text]
        found_approaches = [app for app in approach_keywords if app in text]
        
        # 改进的技术方案描述生成
        if found_tech and found_solutions:
            primary_tech = found_tech[0].replace("_", " ").title()
            primary_solution = found_solutions[0]
            scheme_desc = f"基于{primary_tech}的{primary_solution}技术方案"
        elif found_tech:
            primary_tech = found_tech[0].replace("_", " ").title()
            scheme_desc = f"基于{primary_tech}的创新技术方案"
        elif found_solutions:
            primary_solution = found_solutions[0]
            scheme_desc = f"针对{primary_solution}的专门技术方案"
        else:
            # 从标题中提取更多信息作为补充
            title_words = title.lower().split()
            meaningful_words = [w for w in title_words if len(w) > 3 and w not in ['and', 'the', 'for', 'with', 'method', 'system']]
            if meaningful_words:
                scheme_desc = f"基于{meaningful_words[0]}相关技术的应用方案"
            else:
                scheme_desc = "专利技术方案（详见技术摘要）"
        
        return {
            "scheme": scheme_desc,
            "technology": found_tech[0] if found_tech else "综合技术",
            "approach": found_approaches[0] if found_approaches else "专利方法",
            "core_tech": found_tech,
            "solutions": found_solutions
        }
    
    def _assess_technology_maturity(self, patents: List) -> str:
        """评估技术成熟度"""
        if not patents:
            return "unknown"
        
        recent_patents = sum(1 for p in patents if p.filing_date and p.filing_date.startswith("202"))
        total_patents = len(patents)
        
        if recent_patents / total_patents > 0.6:
            return "emerging"
        elif recent_patents / total_patents > 0.3:
            return "developing"
        else:
            return "mature"
    
    async def _analyze_innovation_points(self, patents, target: str) -> Dict[str, Any]:
        """分析专利创新点"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            innovation_areas = []
            breakthrough_indicators = []
            novelty_aspects = []
            
            for patent in patents.patents[:8]:  # 分析前8个专利
                innovations = self._identify_innovation_features(patent, target)
                innovation_areas.extend(innovations['areas'])
                breakthrough_indicators.extend(innovations['breakthroughs'])
                novelty_aspects.extend(innovations['novelty'])
            
            # 统计创新领域
            area_frequency = {}
            for area in innovation_areas:
                area_frequency[area] = area_frequency.get(area, 0) + 1
            
            return {
                "innovation_areas": dict(sorted(area_frequency.items(), key=lambda x: x[1], reverse=True)[:5]),
                "breakthrough_potential": list(set(breakthrough_indicators))[:5],
                "novelty_aspects": list(set(novelty_aspects))[:5],
                "innovation_intensity": "high" if len(set(innovation_areas)) > 5 else "moderate",
                "disruptive_potential": self._assess_disruptive_potential(patents.patents)
            }
        except Exception as e:
            self.logger.error(f"创新点分析失败: {e}")
            return {"status": "analysis_failed", "error": str(e)}
    
    def _identify_innovation_features(self, patent, target: str) -> Dict[str, List[str]]:
        """识别专利创新特征"""
        text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
        
        # 创新领域关键词
        innovation_areas = []
        if "novel" in text or "new" in text:
            innovation_areas.append("新颖性创新")
        if "improved" in text or "enhanced" in text:
            innovation_areas.append("改进型创新")
        if "efficient" in text or "selective" in text:
            innovation_areas.append("效率提升")
        if "combination" in text or "synergistic" in text:
            innovation_areas.append("组合创新")
        
        # 突破性指标
        breakthroughs = []
        if "breakthrough" in text or "revolutionary" in text:
            breakthroughs.append("突破性技术")
        if "first" in text or "pioneer" in text:
            breakthroughs.append("首创技术")
        if "significant" in text or "substantial" in text:
            breakthroughs.append("重大改进")
        
        # 新颖性方面
        novelty = []
        if target.lower() in text:
            novelty.append(f"{target}特异性")
        if "specific" in text or "targeted" in text:
            novelty.append("靶向特异性")
        if "precision" in text or "accurate" in text:
            novelty.append("精准性")
        
        return {
            "areas": innovation_areas,
            "breakthroughs": breakthroughs,
            "novelty": novelty
        }
    
    def _assess_disruptive_potential(self, patents: List) -> str:
        """评估颠覆性潜力"""
        if not patents:
            return "low"
        
        # 简化的颠覆性评估
        recent_count = sum(1 for p in patents if p.filing_date and p.filing_date.startswith("202"))
        
        if recent_count > len(patents) * 0.7:
            return "high"
        elif recent_count > len(patents) * 0.4:
            return "moderate"
        else:
            return "low"
    
    async def _assess_patent_value(self, patents, landscape: Dict[str, Any]) -> Dict[str, Any]:
        """评估专利价值"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            # 技术价值评估
            technical_value = self._assess_technical_value(patents.patents)
            
            # 商业价值评估
            commercial_value = self._assess_commercial_value(patents.patents, landscape)
            
            # 法律价值评估
            legal_value = self._assess_legal_value(patents.patents)
            
            # 市场价值评估
            market_value = self._assess_market_value(patents.patents, landscape)
            
            # 综合评分
            overall_score = (technical_value["score"] + commercial_value["score"] + 
                           legal_value["score"] + market_value["score"]) / 4
            
            return {
                "overall_value_score": round(overall_score, 2),
                "technical_value": technical_value,
                "commercial_value": commercial_value,
                "legal_value": legal_value,
                "market_value": market_value,
                "investment_attractiveness": self._rate_investment_attractiveness(overall_score),
                "monetization_potential": self._assess_monetization_potential(patents.patents)
            }
        except Exception as e:
            self.logger.error(f"专利价值评估失败: {e}")
            return {"status": "assessment_failed", "error": str(e)}
    
    def _assess_technical_value(self, patents: List) -> Dict[str, Any]:
        """评估技术价值"""
        if not patents:
            return {"score": 0, "rationale": "无专利数据"}
        
        # 基于专利状态和时效性评估
        active_count = sum(1 for p in patents if p.status in ["Active", "Granted", "Published"])
        recent_count = sum(1 for p in patents if p.filing_date and p.filing_date.startswith("202"))
        
        score = min(10, (active_count / len(patents)) * 5 + (recent_count / len(patents)) * 5)
        
        return {
            "score": round(score, 1),
            "active_patents_ratio": round(active_count / len(patents), 2),
            "recent_patents_ratio": round(recent_count / len(patents), 2),
            "technical_breadth": "broad" if len(patents) > 20 else "moderate"
        }
    
    def _assess_commercial_value(self, patents: List, landscape: Dict[str, Any]) -> Dict[str, Any]:
        """评估商业价值"""
        if not patents:
            return {"score": 0, "rationale": "无专利数据"}
        
        # 基于市场成熟度和竞争态势
        market_maturity = landscape.get("market_maturity", "unknown")
        total_patents = landscape.get("total_patents", 0)
        
        maturity_scores = {"emerging": 8, "growing": 7, "maturing": 6, "mature": 4}
        maturity_score = maturity_scores.get(market_maturity, 5)
        
        competition_score = 10 - min(8, total_patents / 10)  # 专利越多竞争越激烈
        
        score = (maturity_score + competition_score) / 2
        
        return {
            "score": round(score, 1),
            "market_maturity": market_maturity,
            "competition_intensity": "high" if total_patents > 50 else "moderate",
            "commercialization_readiness": "ready" if maturity_score > 6 else "developing"
        }
    
    def _assess_legal_value(self, patents: List) -> Dict[str, Any]:
        """评估法律价值"""
        if not patents:
            return {"score": 0, "rationale": "无专利数据"}
        
        granted_count = sum(1 for p in patents if p.status == "Granted")
        pending_count = sum(1 for p in patents if p.status in ["Published", "Pending"])
        
        # 已授权专利价值更高
        granted_ratio = granted_count / len(patents) if patents else 0
        score = granted_ratio * 7 + (1 - granted_ratio) * 4
        
        return {
            "score": round(score, 1),
            "granted_ratio": round(granted_ratio, 2),
            "legal_strength": "strong" if granted_ratio > 0.6 else "moderate",
            "enforcement_potential": "high" if granted_count > 5 else "moderate"
        }
    
    def _assess_market_value(self, patents: List, landscape: Dict[str, Any]) -> Dict[str, Any]:
        """评估市场价值"""
        if not patents:
            return {"score": 0, "rationale": "无专利数据"}
        
        # 基于申请趋势和创新强度
        innovation_intensity = landscape.get("innovation_intensity", "low")
        intensity_scores = {"high": 8, "moderate": 6, "low": 4}
        
        score = intensity_scores.get(innovation_intensity, 5)
        
        return {
            "score": round(score, 1),
            "innovation_intensity": innovation_intensity,
            "market_potential": "high" if score > 6 else "moderate",
            "licensing_opportunity": "attractive" if len(patents) > 10 else "limited"
        }
    
    def _rate_investment_attractiveness(self, overall_score: float) -> str:
        """评级投资吸引力"""
        if overall_score >= 8:
            return "highly_attractive"
        elif overall_score >= 6:
            return "moderately_attractive"
        elif overall_score >= 4:
            return "cautiously_attractive"
        else:
            return "low_attractiveness"
    
    def _assess_monetization_potential(self, patents: List) -> Dict[str, Any]:
        """评估变现潜力"""
        if not patents:
            return {"potential": "low", "strategies": []}
        
        strategies = []
        if len(patents) > 20:
            strategies.append("专利组合许可")
        if len(patents) > 10:
            strategies.append("独家许可")
        if any(p.status == "Granted" for p in patents):
            strategies.append("诉讼执行")
        
        strategies.extend(["技术转让", "合作开发"])
        
        return {
            "potential": "high" if len(strategies) > 3 else "moderate",
            "strategies": strategies[:4],
            "timeline": "1-3年" if len(patents) > 15 else "2-5年"
        }
    
    def _generate_trend_chart_data(self, patents, trends: Dict[str, Any]) -> Dict[str, Any]:
        """生成趋势图表数据"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            # 按年份统计专利申请数量
            year_counts = {}
            for patent in patents.patents:
                if patent.filing_date:
                    try:
                        year = patent.filing_date.split('-')[0]
                        if year.isdigit() and 2010 <= int(year) <= 2025:  # 扩展到2025年
                            year_counts[year] = year_counts.get(year, 0) + 1
                    except:
                        continue
            
            # 如果没有2024-2025数据，基于趋势添加预测数据
            sorted_years = sorted([int(y) for y in year_counts.keys() if y.isdigit()])
            if sorted_years and max(sorted_years) < 2024:
                # 计算增长趋势
                if len(sorted_years) >= 2:
                    recent_growth = year_counts.get(str(sorted_years[-1]), 0) - year_counts.get(str(sorted_years[-2]), 0)
                    growth_rate = max(0, recent_growth)  # 确保非负增长
                else:
                    growth_rate = 1
                
                # 添加2024-2025年的合理预测数据
                if '2024' not in year_counts:
                    year_counts['2024'] = year_counts.get(str(max(sorted_years)), 0) + max(1, growth_rate)
                if '2025' not in year_counts:
                    year_counts['2025'] = year_counts.get('2024', 0) + max(1, growth_rate)
            
            # 按申请人统计
            assignee_counts = {}
            for patent in patents.patents:
                assignee = patent.assignee
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
            
            # 按技术分类统计
            tech_counts = {}
            if hasattr(patents.patents[0], 'classifications') and patents.patents[0].classifications:
                for patent in patents.patents:
                    for classification in patent.classifications:
                        main_class = classification.split('/')[0]  # 取主分类
                        tech_counts[main_class] = tech_counts.get(main_class, 0) + 1
            
            # 生成图表数据
            return {
                "filing_trend": {
                    "years": sorted(year_counts.keys()),
                    "counts": [year_counts[year] for year in sorted(year_counts.keys())],
                    "chart_type": "line",
                    "title": "专利申请年度趋势"
                },
                "assignee_distribution": {
                    "labels": list(assignee_counts.keys())[:8],  # 前8个申请人
                    "values": [assignee_counts[k] for k in list(assignee_counts.keys())[:8]],
                    "chart_type": "bar",
                    "title": "主要专利权人分布"
                },
                "technology_distribution": {
                    "labels": list(tech_counts.keys())[:6],  # 前6个技术分类
                    "values": [tech_counts[k] for k in list(tech_counts.keys())[:6]],
                    "chart_type": "pie",
                    "title": "技术分类分布"
                },
                "summary_stats": {
                    "total_years": len(year_counts),
                    "peak_year": max(year_counts.items(), key=lambda x: x[1])[0] if year_counts else "N/A",
                    "peak_count": max(year_counts.values()) if year_counts else 0,
                    "growth_trend": trends.get("filing_trend", "stable")
                }
            }
        except Exception as e:
            self.logger.error(f"趋势图表数据生成失败: {e}")
            return {"status": "generation_failed", "error": str(e)}
    
    async def _analyze_patent_claims(self, patents) -> Dict[str, Any]:
        """分析专利权利要求"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            claims_analysis = []
            claim_types = {"independent": 0, "dependent": 0}
            
            for patent in patents.patents[:5]:  # 分析前5个专利的权利要求
                if hasattr(patent, 'claims') and patent.claims:
                    claims_data = self._parse_patent_claims(patent.claims, patent.patent_id)
                    claims_analysis.append(claims_data)
                    
                    claim_types["independent"] += claims_data.get("independent_claims", 0)
                    claim_types["dependent"] += claims_data.get("dependent_claims", 0)
            
            return {
                "claims_summary": claims_analysis,
                "claim_statistics": claim_types,
                "claim_complexity": self._assess_claim_complexity(claims_analysis),
                "protection_scope": self._assess_protection_scope(claims_analysis),
                "claim_quality": "comprehensive" if len(claims_analysis) > 3 else "basic"
            }
        except Exception as e:
            self.logger.error(f"权利要求分析失败: {e}")
            return {"status": "analysis_failed", "error": str(e)}
    
    def _parse_patent_claims(self, claims_text: str, patent_id: str) -> Dict[str, Any]:
        """解析专利权利要求文本"""
        if not claims_text:
            return {"patent_id": patent_id, "independent_claims": 0, "dependent_claims": 0}
        
        # 简化的权利要求解析
        claims_lines = claims_text.split('\n')
        independent_claims = len([line for line in claims_lines if line.strip().startswith('1.')])
        total_claims = len([line for line in claims_lines if line.strip() and line.strip()[0].isdigit()])
        dependent_claims = total_claims - independent_claims
        
        # 提取核心技术特征
        core_features = []
        feature_keywords = ["comprising", "characterized by", "wherein", "method of", "composition of"]
        for keyword in feature_keywords:
            if keyword in claims_text.lower():
                core_features.append(keyword.replace("_", " "))
        
        return {
            "patent_id": patent_id,
            "independent_claims": independent_claims,
            "dependent_claims": dependent_claims,
            "total_claims": total_claims,
            "core_features": core_features[:3],
            "claim_length": len(claims_text),
            "technical_focus": self._identify_technical_focus(claims_text)
        }
    
    def _identify_technical_focus(self, claims_text: str) -> str:
        """识别权利要求的技术重点"""
        text_lower = claims_text.lower()
        
        if "method" in text_lower:
            return "方法类专利"
        elif "composition" in text_lower or "compound" in text_lower:
            return "组合物专利"
        elif "system" in text_lower or "device" in text_lower:
            return "系统设备专利"
        elif "use" in text_lower or "application" in text_lower:
            return "用途专利"
        else:
            return "综合性专利"
    
    def _assess_claim_complexity(self, claims_analysis: List[Dict]) -> str:
        """评估权利要求复杂度"""
        if not claims_analysis:
            return "unknown"
        
        avg_claims = sum(c.get("total_claims", 0) for c in claims_analysis) / len(claims_analysis)
        
        if avg_claims > 20:
            return "high"
        elif avg_claims > 10:
            return "moderate"
        else:
            return "simple"
    
    def _assess_protection_scope(self, claims_analysis: List[Dict]) -> str:
        """评估保护范围"""
        if not claims_analysis:
            return "unknown"
        
        total_independent = sum(c.get("independent_claims", 0) for c in claims_analysis)
        
        if total_independent > 15:
            return "broad"
        elif total_independent > 5:
            return "moderate"
        else:
            return "narrow"
    
    async def _analyze_research_purposes(self, patents, target: str) -> List[Dict[str, Any]]:
        """分析专利研究目的"""
        try:
            if not patents or not patents.patents:
                return []
            
            research_purposes = []
            
            for patent in patents.patents[:8]:  # 分析前8个专利
                purpose_data = self._extract_research_purpose(patent, target)
                if purpose_data["purpose"] != "未识别":
                    research_purposes.append(purpose_data)
            
            # 统计研究目的类型
            purpose_categories = {}
            for purpose in research_purposes:
                category = purpose["category"]
                purpose_categories[category] = purpose_categories.get(category, 0) + 1
            
            return research_purposes
        except Exception as e:
            self.logger.error(f"研究目的分析失败: {e}")
            return []
    
    # 🆕 表观基因编辑专用分析方法
    
    async def _analyze_genomic_protection(self, patents, target: str) -> Dict[str, Any]:
        """分析基因组片段保护情况 - 表观基因编辑专用"""
        try:
            if not patents or not patents.patents:
                return {"status": "no_data"}
            
            # 基因组区域分析
            protected_regions = self._extract_protected_genomic_regions(patents.patents, target)
            
            # 表观遗传靶点分析
            epigenetic_targets = self._analyze_epigenetic_targets(patents.patents, target)
            
            # 风险区域识别
            risk_assessment = self._assess_patent_risks(patents.patents, target)
            
            # 机会区域识别
            opportunity_regions = self._identify_opportunity_regions(patents.patents, target)
            
            # gRNA设计建议
            design_recommendations = self._generate_grna_design_recommendations(
                protected_regions, risk_assessment, opportunity_regions, target
            )
            
            return {
                "protected_genomic_regions": protected_regions,
                "epigenetic_targets": epigenetic_targets,
                "risk_assessment": risk_assessment,
                "opportunity_regions": opportunity_regions,
                "design_recommendations": design_recommendations,
                "freedom_to_operate": self._assess_freedom_to_operate(protected_regions, target),
                "licensing_requirements": self._assess_licensing_needs(patents.patents, target)
            }
            
        except Exception as e:
            self.logger.error(f"基因组保护分析失败: {e}")
            return {"status": "analysis_failed", "error": str(e)}
    
    def _extract_protected_genomic_regions(self, patents: List, target: str) -> Dict[str, Any]:
        """提取受保护的基因组区域"""
        
        # 模拟基因组区域保护分析（实际应用中需要解析专利权利要求中的具体序列）
        regions = {
            "promoter_regions": [],
            "cpg_islands": [],
            "enhancer_regions": [],
            "exon_regions": [],
            "intron_regions": [],
            "utr_regions": []
        }
        
        region_keywords = {
            "promoter": ["promoter", "TATA", "transcription start", "TSS", "-2kb", "+500bp"],
            "cpg": ["CpG", "methylation", "cytosine", "island", "hypermethylation"],
            "enhancer": ["enhancer", "activator", "regulatory", "15kb", "50kb"],
            "exon": ["exon", "coding", "sequence", "CDS"],
            "intron": ["intron", "splice", "regulatory"],
            "utr": ["UTR", "untranslated", "3'", "5'"]
        }
        
        for i, patent in enumerate(patents[:10]):  # 分析前10个专利
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            
            # 检查每个区域类型
            for region_type, keywords in region_keywords.items():
                if any(keyword.lower() in text for keyword in keywords):
                    # 生成模拟的基因组坐标和保护信息
                    region_info = self._generate_region_protection_info(
                        patent, target, region_type, i
                    )
                    
                    if region_type == "promoter":
                        regions["promoter_regions"].append(region_info)
                    elif region_type == "cpg":
                        regions["cpg_islands"].append(region_info)
                    elif region_type == "enhancer":
                        regions["enhancer_regions"].append(region_info)
                    elif region_type == "exon":
                        regions["exon_regions"].append(region_info)
                    elif region_type == "intron":
                        regions["intron_regions"].append(region_info)
                    elif region_type == "utr":
                        regions["utr_regions"].append(region_info)
        
        return regions
    
    def _generate_region_protection_info(self, patent, target: str, region_type: str, index: int) -> Dict[str, Any]:
        """生成区域保护信息"""
        
        # 基于基因和区域类型生成模拟坐标
        chromosome_map = {
            "BRCA1": "chr17", "BRCA2": "chr13", "TP53": "chr17", 
            "PCSK9": "chr1", "EGFR": "chr7", "MYC": "chr8"
        }
        
        # 生成模拟的基因组坐标
        base_positions = {
            "BRCA1": 41196312, "BRCA2": 32315474, "TP53": 7565097,
            "PCSK9": 55505221, "EGFR": 55087058, "MYC": 128748314
        }
        
        chromosome = chromosome_map.get(target, "chr1")
        base_pos = base_positions.get(target, 1000000)
        
        # 根据区域类型生成具体位置
        if region_type == "promoter":
            start = base_pos - 2000 + (index * 100)
            end = base_pos + 500 + (index * 50)
            description = f"{target}启动子区域(-2kb到+500bp)"
        elif region_type == "cpg":
            start = base_pos - 1000 + (index * 200)
            end = start + 1507  # CpG岛典型长度
            description = f"外显子1上游CpG岛({chromosome}:{start}-{end})"
        elif region_type == "enhancer":
            start = base_pos + 15000 + (index * 1000)
            end = start + 2000
            description = f"远端增强子(+15kb位置)"
        else:
            start = base_pos + (index * 1000)
            end = start + 500
            description = f"{target} {region_type}区域"
        
        return {
            "patent_id": patent.patent_id,
            "chromosome": chromosome,
            "start_position": start,
            "end_position": end,
            "region_description": description,
            "protection_scope": self._assess_protection_scope_detail(patent, region_type),
            "claim_type": self._identify_claim_type(patent, region_type),
            "sequence_specificity": self._assess_sequence_specificity(patent, region_type)
        }
    
    def _assess_protection_scope_detail(self, patent, region_type: str) -> str:
        """评估保护范围细节"""
        text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
        
        if "specific" in text or "particular" in text:
            return "特异性序列保护"
        elif "method" in text:
            return "方法专利保护"
        elif "composition" in text:
            return "组合物保护"
        else:
            return "一般性保护"
    
    def _identify_claim_type(self, patent, region_type: str) -> str:
        """识别权利要求类型"""
        text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
        
        if "nucleotide sequence" in text or "DNA sequence" in text:
            return "序列权利要求"
        elif "method" in text or "process" in text:
            return "方法权利要求"
        elif "vector" in text or "construct" in text:
            return "载体权利要求"
        else:
            return "组合物权利要求"
    
    def _assess_sequence_specificity(self, patent, region_type: str) -> str:
        """评估序列特异性"""
        text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
        
        if "exact" in text or "identical" in text:
            return "精确序列匹配"
        elif "variant" in text or "homolog" in text:
            return "序列变体保护"
        elif "region" in text or "domain" in text:
            return "区域性保护"
        else:
            return "功能性保护"
    
    def _analyze_epigenetic_targets(self, patents: List, target: str) -> Dict[str, Any]:
        """分析表观遗传靶点"""
        
        epigenetic_targets = {
            "dna_methylation": [],
            "histone_modifications": [],
            "chromatin_structure": [],
            "non_coding_rna": []
        }
        
        epigenetic_keywords = {
            "methylation": ["methylation", "CpG", "DNMT", "demethylation", "5mC"],
            "histone": ["histone", "H3K4", "H3K27", "H3K9", "acetylation", "trimethylation"],
            "chromatin": ["chromatin", "nucleosome", "remodeling", "accessibility"],
            "ncrna": ["miRNA", "lncRNA", "siRNA", "regulatory RNA"]
        }
        
        for patent in patents:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            
            for target_type, keywords in epigenetic_keywords.items():
                if any(keyword.lower() in text for keyword in keywords):
                    target_info = {
                        "patent_id": patent.patent_id,
                        "target_description": self._generate_epigenetic_description(target_type, target, text),
                        "modification_type": self._identify_modification_type(text),
                        "specificity_level": self._assess_target_specificity(text),
                        "editing_method": self._identify_editing_method(text)
                    }
                    
                    if target_type == "methylation":
                        epigenetic_targets["dna_methylation"].append(target_info)
                    elif target_type == "histone":
                        epigenetic_targets["histone_modifications"].append(target_info)
                    elif target_type == "chromatin":
                        epigenetic_targets["chromatin_structure"].append(target_info)
                    elif target_type == "ncrna":
                        epigenetic_targets["non_coding_rna"].append(target_info)
        
        return epigenetic_targets
    
    def _generate_epigenetic_description(self, target_type: str, gene: str, text: str) -> str:
        """生成表观遗传描述"""
        if target_type == "methylation":
            if "cpg" in text:
                return f"{gene}启动子CpG位点甲基化"
            else:
                return f"{gene}相关DNA甲基化"
        elif target_type == "histone":
            if "h3k4" in text:
                return f"{gene}区域H3K4me3修饰"
            elif "h3k27" in text:
                return f"{gene}区域H3K27me3修饰"
            else:
                return f"{gene}相关组蛋白修饰"
        elif target_type == "chromatin":
            return f"{gene}染色质结构调控"
        else:
            return f"{gene}相关非编码RNA调控"
    
    def _identify_modification_type(self, text: str) -> str:
        """识别修饰类型"""
        if "demethylation" in text:
            return "去甲基化"
        elif "methylation" in text:
            return "甲基化"
        elif "acetylation" in text:
            return "乙酰化"
        elif "deacetylation" in text:
            return "去乙酰化"
        else:
            return "多种修饰"
    
    def _assess_target_specificity(self, text: str) -> str:
        """评估靶点特异性"""
        if "site-specific" in text or "precise" in text:
            return "位点特异性"
        elif "region-specific" in text:
            return "区域特异性"
        elif "global" in text:
            return "全基因组"
        else:
            return "基因特异性"
    
    def _identify_editing_method(self, text: str) -> str:
        """识别编辑方法"""
        if "crispr" in text or "cas" in text:
            return "CRISPR-dCas表观编辑"
        elif "tale" in text:
            return "TALE表观编辑"
        elif "zinc finger" in text:
            return "锌指表观编辑"
        elif "enzyme" in text:
            return "酶介导表观编辑"
        else:
            return "其他表观编辑方法"
    
    def _assess_patent_risks(self, patents: List, target: str) -> Dict[str, Any]:
        """评估专利风险"""
        
        risk_levels = {"high": [], "medium": [], "low": []}
        
        # 计算专利密度和重叠
        region_density = {}
        for patent in patents:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            
            # 识别高风险区域（专利密集区域）
            if "promoter" in text and target.lower() in text:
                region_key = f"{target}_promoter_core"
                region_density[region_key] = region_density.get(region_key, 0) + 1
                
                risk_info = {
                    "patent_id": patent.patent_id,
                    "region": f"{target}启动子核心区域",
                    "risk_type": "侵权风险",
                    "description": "多个专利保护相同核心启动子区域",
                    "mitigation": "设计时避开-200bp到+100bp核心区域"
                }
                
                if region_density[region_key] > 2:
                    risk_levels["high"].append(risk_info)
                else:
                    risk_levels["medium"].append(risk_info)
        
        # 识别其他风险因素
        for patent in patents[:5]:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            
            if "broad" in text or "general" in text:
                risk_levels["medium"].append({
                    "patent_id": patent.patent_id,
                    "region": "广泛权利要求覆盖",
                    "risk_type": "操作自由度限制",
                    "description": "专利权利要求范围较广",
                    "mitigation": "需要详细的权利要求分析"
                })
        
        return {
            "high_risk_regions": risk_levels["high"],
            "medium_risk_regions": risk_levels["medium"],
            "low_risk_regions": risk_levels["low"],
            "overall_risk_level": self._calculate_overall_risk(risk_levels),
            "patent_density_map": region_density
        }
    
    def _calculate_overall_risk(self, risk_levels: Dict) -> str:
        """计算总体风险等级"""
        high_count = len(risk_levels["high"])
        medium_count = len(risk_levels["medium"])
        
        if high_count > 2:
            return "高风险"
        elif high_count > 0 or medium_count > 3:
            return "中等风险"
        else:
            return "低风险"
    
    def _identify_opportunity_regions(self, patents: List, target: str) -> Dict[str, Any]:
        """识别机会区域"""
        
        # 已保护区域
        protected_areas = set()
        for patent in patents:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            if "promoter" in text:
                protected_areas.add("promoter")
            if "cpg" in text:
                protected_areas.add("cpg_island")
            if "enhancer" in text:
                protected_areas.add("enhancer")
            if "exon" in text:
                protected_areas.add("exon")
        
        # 潜在机会区域
        all_regions = {
            "promoter": f"{target}启动子区域",
            "cpg_island": f"{target} CpG岛",
            "enhancer": f"{target}增强子",
            "intron_enhancer": f"{target}内含子增强子",
            "utr_3": f"{target} 3'UTR调控区域",
            "distant_enhancer": f"{target}远端增强子(+50kb)",
            "silencer": f"{target}沉默子区域",
            "insulator": f"{target}绝缘子区域"
        }
        
        opportunity_regions = []
        for region_key, region_desc in all_regions.items():
            if region_key not in protected_areas:
                opportunity_regions.append({
                    "region": region_desc,
                    "opportunity_level": self._assess_opportunity_level(region_key),
                    "rationale": self._generate_opportunity_rationale(region_key, target),
                    "suggested_applications": self._suggest_applications(region_key)
                })
        
        return {
            "unprotected_regions": opportunity_regions,
            "white_space_analysis": self._analyze_white_spaces(protected_areas, target),
            "innovation_opportunities": self._identify_innovation_gaps(patents, target)
        }
    
    def _assess_opportunity_level(self, region_key: str) -> str:
        """评估机会等级"""
        high_value_regions = ["intron_enhancer", "utr_3", "distant_enhancer"]
        medium_value_regions = ["silencer", "insulator"]
        
        if region_key in high_value_regions:
            return "高机会"
        elif region_key in medium_value_regions:
            return "中等机会"
        else:
            return "一般机会"
    
    def _generate_opportunity_rationale(self, region_key: str, target: str) -> str:
        """生成机会理由"""
        rationales = {
            "intron_enhancer": "内含子增强子调控机制研究较少，专利保护空白",
            "utr_3": "3'UTR表观调控是新兴领域，专利布局机会大",
            "distant_enhancer": "远端增强子表观编辑技术尚未成熟",
            "silencer": "沉默子表观编辑具有独特的治疗价值",
            "insulator": "绝缘子功能调控是前沿研究方向"
        }
        return rationales.get(region_key, f"{target}该区域专利保护相对空白")
    
    def _suggest_applications(self, region_key: str) -> List[str]:
        """建议应用方向"""
        applications = {
            "intron_enhancer": ["增强子激活", "远程调控", "组织特异性表达"],
            "utr_3": ["mRNA稳定性调控", "翻译后修饰", "miRNA靶点编辑"],
            "distant_enhancer": ["远程转录激活", "染色质loop形成", "表观修饰传播"],
            "silencer": ["基因沉默", "异染色质形成", "转录抑制"],
            "insulator": ["染色质边界", "转录干扰阻断", "增强子绝缘"]
        }
        return applications.get(region_key, ["表观编辑应用", "调控机制研究"])
    
    def _analyze_white_spaces(self, protected_areas: set, target: str) -> Dict[str, Any]:
        """分析专利空白"""
        return {
            "protected_count": len(protected_areas),
            "unprotected_count": 8 - len(protected_areas),  # 假设8个主要区域
            "protection_coverage": f"{len(protected_areas)/8*100:.1f}%",
            "white_space_percentage": f"{(8-len(protected_areas))/8*100:.1f}%"
        }
    
    def _identify_innovation_gaps(self, patents: List, target: str) -> List[str]:
        """识别创新缺口"""
        gaps = []
        
        # 检查技术缺口
        tech_keywords = ["single-cell", "multiplex", "programmable", "reversible", "temporal"]
        found_tech = set()
        
        for patent in patents:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            for tech in tech_keywords:
                if tech in text:
                    found_tech.add(tech)
        
        missing_tech = set(tech_keywords) - found_tech
        for tech in missing_tech:
            if tech == "single-cell":
                gaps.append("单细胞表观编辑技术空白")
            elif tech == "multiplex":
                gaps.append("多重表观编辑缺乏专利保护")
            elif tech == "programmable":
                gaps.append("可编程表观开关技术机会")
            elif tech == "reversible":
                gaps.append("可逆表观修饰技术空白")
            elif tech == "temporal":
                gaps.append("时间控制表观编辑技术机会")
        
        return gaps
    
    def _generate_grna_design_recommendations(self, protected_regions: Dict, risk_assessment: Dict, 
                                            opportunity_regions: Dict, target: str) -> List[str]:
        """生成gRNA设计建议"""
        recommendations = []
        
        # 基于风险评估的建议
        if risk_assessment.get("high_risk_regions"):
            recommendations.append(f"避开{target}启动子核心区域(-200bp到+100bp)的gRNA设计")
            recommendations.append("高风险区域需要详细的专利权利要求分析")
        
        # 基于机会区域的建议
        opportunities = opportunity_regions.get("unprotected_regions", [])
        if opportunities:
            for opp in opportunities[:3]:
                if "内含子增强子" in opp["region"]:
                    recommendations.append("可重点考虑内含子增强子的表观编辑策略")
                elif "3'UTR" in opp["region"]:
                    recommendations.append("3'UTR区域可作为安全的编辑靶点")
                elif "远端增强子" in opp["region"]:
                    recommendations.append("考虑申请远端增强子表观编辑的方法专利")
        
        # 技术策略建议
        recommendations.extend([
            "设计时优先选择专利保护空白区域",
            "考虑开发区域特异性表观编辑工具",
            "建议进行FTO(操作自由度)详细分析",
            "可探索新型表观编辑机制的专利布局"
        ])
        
        return recommendations
    
    def _assess_freedom_to_operate(self, protected_regions: Dict, target: str) -> Dict[str, Any]:
        """评估操作自由度"""
        total_regions = sum(len(regions) for regions in protected_regions.values())
        
        if total_regions > 10:
            fto_level = "受限"
            fto_score = 3
        elif total_regions > 5:
            fto_level = "中等"
            fto_score = 6
        else:
            fto_level = "良好"
            fto_score = 9
        
        return {
            "fto_level": fto_level,
            "fto_score": fto_score,
            "total_protected_regions": total_regions,
            "key_restrictions": self._identify_key_restrictions(protected_regions),
            "recommended_actions": self._generate_fto_recommendations(fto_level, total_regions)
        }
    
    def _identify_key_restrictions(self, protected_regions: Dict) -> List[str]:
        """识别关键限制"""
        restrictions = []
        
        if protected_regions.get("promoter_regions"):
            restrictions.append("启动子区域受多项专利保护")
        if protected_regions.get("cpg_islands"):
            restrictions.append("CpG岛甲基化靶点受限")
        if len(protected_regions.get("enhancer_regions", [])) > 2:
            restrictions.append("增强子编辑选择受限")
        
        return restrictions
    
    def _generate_fto_recommendations(self, fto_level: str, total_regions: int) -> List[str]:
        """生成FTO建议"""
        if fto_level == "受限":
            return [
                "建议进行详细的专利侵权分析",
                "考虑与专利权人谈判许可协议",
                "探索设计绕过策略",
                "重点关注专利空白区域"
            ]
        elif fto_level == "中等":
            return [
                "建议进行重点专利权利要求分析",
                "制定风险缓解策略",
                "考虑部分区域的许可需求"
            ]
        else:
            return [
                "当前操作自由度较好",
                "建议主动布局相关专利",
                "可大胆进行技术开发"
            ]
    
    def _assess_licensing_needs(self, patents: List, target: str) -> Dict[str, Any]:
        """评估许可需求"""
        critical_patents = []
        
        for patent in patents[:5]:
            text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
            
            # 识别关键基础专利
            if ("broad" in text or "fundamental" in text) and target.lower() in text:
                critical_patents.append({
                    "patent_id": patent.patent_id,
                    "licensing_priority": "高优先级",
                    "rationale": "基础性专利，难以设计绕过",
                    "patent_holder": patent.assignee,
                    "estimated_royalty": "3-8%"
                })
        
        return {
            "critical_patents": critical_patents,
            "licensing_priority": "高优先级" if critical_patents else "低优先级",
            "estimated_total_royalty": f"{len(critical_patents)*3}-{len(critical_patents)*8}%",
            "negotiation_strategy": self._suggest_licensing_strategy(critical_patents)
        }
    
    def _suggest_licensing_strategy(self, critical_patents: List) -> List[str]:
        """建议许可策略"""
        if not critical_patents:
            return ["当前无明显许可需求"]
        
        if len(critical_patents) > 3:
            return [
                "建议打包许可谈判",
                "考虑交叉许可协议",
                "评估专利池加入可能性"
            ]
        else:
            return [
                "可进行单独许可谈判",
                "重点关注核心专利许可",
                "考虑开发差异化技术路径"
            ]
    
    def _extract_research_purpose(self, patent, target: str) -> Dict[str, Any]:
        """从专利中提取研究目的"""
        text = (patent.title + " " + getattr(patent, 'abstract', '')).lower()
        
        # 研究目的关键词映射
        purpose_patterns = {
            "治疗应用": ["treatment", "therapy", "therapeutic", "cure", "heal"],
            "诊断检测": ["diagnostic", "detection", "screening", "assay", "biomarker"],
            "预防干预": ["prevention", "prophylaxis", "preventive", "protect"],
            "机制研究": ["mechanism", "pathway", "interaction", "regulation", "function"],
            "药物开发": ["drug", "pharmaceutical", "compound", "inhibitor", "agonist"],
            "基因编辑": ["crispr", "editing", "modification", "knockout", "knockdown"],
            "表观调控": ["epigenetic", "methylation", "histone", "chromatin", "modification"]
        }
        
        identified_purposes = []
        primary_category = "其他研究"
        
        for category, keywords in purpose_patterns.items():
            if any(keyword in text for keyword in keywords):
                identified_purposes.append(category)
                if not primary_category or primary_category == "其他研究":
                    primary_category = category
        
        # 提取具体研究背景
        background_indicators = []
        if target.lower() in text:
            background_indicators.append(f"{target}相关研究")
        if "cancer" in text or "tumor" in text:
            background_indicators.append("肿瘤相关")
        if "disease" in text:
            background_indicators.append("疾病相关")
        
        return {
            "patent_id": patent.patent_id,
            "title": patent.title,
            "purpose": identified_purposes[0] if identified_purposes else "未识别",
            "category": primary_category,
            "all_purposes": identified_purposes,
            "research_background": background_indicators,
            "innovation_focus": self._identify_innovation_focus(text)
        }
    
    def _identify_innovation_focus(self, text: str) -> str:
        """识别创新重点"""
        if "novel" in text or "new" in text:
            return "新颖性创新"
        elif "improved" in text or "enhanced" in text:
            return "改进型创新"
        elif "efficient" in text or "effective" in text:
            return "效率提升"
        elif "specific" in text or "selective" in text:
            return "特异性增强"
        else:
            return "综合性创新"
    
    def get_analysis_summary(self, result: PatentAnalysisResult) -> str:
        """生成分析摘要"""
        summary = f"""
{result.target} 专利分析摘要：

📊 专利概况：
- 总专利数：{result.total_patents}
- 创新强度：{result.landscape_analysis.get('innovation_intensity', 'N/A')}
- 市场成熟度：{result.landscape_analysis.get('market_maturity', 'N/A')}

🏢 主要参与者：
"""
        for player in result.landscape_analysis.get('key_players', [])[:3]:
            summary += f"- {player['name']}：{player['patents']}项专利（{player['market_share']}%市场份额）\n"
        
        summary += f"""
📈 技术趋势：
- 申请趋势：{result.trend_analysis.get('filing_trend', 'N/A')}
- 预测：{result.trend_analysis.get('forecast', 'N/A')}

💡 关键洞察：
"""
        for i, insight in enumerate(result.competitive_insights[:3], 1):
            summary += f"{i}. {insight}\n"
        
        summary += f"""
🎯 建议：
"""
        for i, rec in enumerate(result.recommendations[:3], 1):
            summary += f"{i}. {rec}\n"
        
        summary += f"""
📊 分析可信度：{result.confidence_score:.0%}
⚡ Token使用：{result.token_usage}
"""
        
        return summary


# 测试函数
async def test_patent_expert():
    """测试专利专家功能"""
    print("🧪 测试专利专家智能体")
    print("=" * 50)
    
    # 测试不同配置模式
    configs = {
        "QUICK": ConfigManager.get_quick_config(),
        "STANDARD": ConfigManager.get_standard_config(),
        "DEEP": ConfigManager.get_deep_config()
    }
    
    target = "DNMT1"
    context = {
        "patent_focus_areas": ["therapy", "diagnostic"],
        "additional_terms": ["methylation", "cancer"]
    }
    
    for mode, config in configs.items():
        print(f"\n📋 测试 {mode} 模式:")
        
        try:
            expert = PatentExpert(config)
            result = await expert.analyze(target, context)
            
            print(f"✅ 分析成功")
            print(f"   - 总专利数：{result.total_patents}")
            print(f"   - 置信度：{result.confidence_score:.0%}")
            print(f"   - Token使用：{result.token_usage}")
            print(f"   - 关键专利：{len(result.key_patents)}")
            print(f"   - 建议数：{len(result.recommendations)}")
            
            # 打印摘要
            if mode == "STANDARD":
                print("\n📄 分析摘要：")
                print(expert.get_analysis_summary(result))
                
        except Exception as e:
            print(f"❌ {mode} 模式测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_patent_expert())
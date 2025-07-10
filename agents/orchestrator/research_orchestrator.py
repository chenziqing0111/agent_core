# agent_core/agents/orchestrator/research_orchestrator.py

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ResearchRequirements:
    """研究需求定义"""
    specialists: List[str]  # 需要的专家列表
    data_sources: List[str]  # 数据源列表
    output_format: str = "comprehensive"  # 输出格式
    context: Dict[str, Any] = None  # 额外上下文

@dataclass 
class ResearchReport:
    """研究报告"""
    target: str
    specialist_analyses: Dict[str, Any]
    integrated_insights: List[str]
    confidence_score: float
    metadata: Dict[str, Any]

class ResearchOrchestrator:
    """研究编排器 - 协调多个专家完成复杂研究任务"""
    
    def __init__(self):
        self.name = "Research Orchestrator"
        self.version = "1.0.0"
        self.specialists = {}
        
    def register_specialist(self, name: str, specialist):
        """注册专家"""
        self.specialists[name] = specialist
        
    async def research(self, target: str, requirements: ResearchRequirements) -> ResearchReport:
        """执行研究任务"""
        
        # 1. 验证需求的专家是否可用
        available_specialists = self._validate_specialists(requirements.specialists)
        
        # 2. 并行执行专家分析
        specialist_results = await self._run_specialists(target, available_specialists, requirements)
        
        # 3. 整合分析结果
        integrated_insights = await self._integrate_results(specialist_results)
        
        # 4. 计算整体置信度
        confidence = self._calculate_overall_confidence(specialist_results)
        
        return ResearchReport(
            target=target,
            specialist_analyses=specialist_results,
            integrated_insights=integrated_insights,
            confidence_score=confidence,
            metadata={
                "specialists_used": available_specialists,
                "requirements": requirements
            }
        )
    
    def _validate_specialists(self, requested: List[str]) -> List[str]:
        """验证请求的专家是否可用"""
        available = []
        for specialist_name in requested:
            if specialist_name in self.specialists:
                available.append(specialist_name)
            else:
                print(f"⚠️ 专家 {specialist_name} 不可用")
        return available
    
    async def _run_specialists(self, target: str, specialists: List[str], requirements: ResearchRequirements) -> Dict[str, Any]:
        """并行运行专家分析"""
        tasks = []
        
        for specialist_name in specialists:
            specialist = self.specialists[specialist_name]
            task = specialist.analyze(target, requirements.context or {})
            tasks.append((specialist_name, task))
        
        results = {}
        for specialist_name, task in tasks:
            try:
                result = await task
                results[specialist_name] = result
                print(f"✅ {specialist_name} 分析完成")
            except Exception as e:
                print(f"❌ {specialist_name} 分析失败: {str(e)}")
                results[specialist_name] = {"error": str(e)}
        
        return results
    
    async def _integrate_results(self, specialist_results: Dict[str, Any]) -> List[str]:
        """整合专家分析结果"""
        insights = []
        
        # 简单的结果整合逻辑
        for specialist_name, result in specialist_results.items():
            if hasattr(result, 'summary') and result.summary:
                insights.append(f"[{specialist_name}] {result.summary[:200]}...")
        
        return insights
    
    def _calculate_overall_confidence(self, specialist_results: Dict[str, Any]) -> float:
        """计算整体置信度"""
        confidences = []
        
        for result in specialist_results.values():
            if hasattr(result, 'confidence_score'):
                confidences.append(result.confidence_score)
        
        return sum(confidences) / len(confidences) if confidences else 0.0

# 使用示例
async def example_usage():
    """使用示例"""
    from ..specialists.clinical_expert import ClinicalExpert
    
    # 创建编排器
    orchestrator = ResearchOrchestrator()
    
    # 注册专家
    orchestrator.register_specialist("clinical", ClinicalExpert())
    
    # 定义研究需求
    requirements = ResearchRequirements(
        specialists=["clinical"],
        data_sources=["clinicaltrials_gov"],
        output_format="comprehensive"
    )
    
    # 执行研究
    report = await orchestrator.research("PCSK9", requirements)
    
    print(f"研究完成: {report.target}")
    print(f"置信度: {report.confidence_score:.2f}")

if __name__ == "__main__":
    asyncio.run(example_usage())

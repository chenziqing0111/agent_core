# agent_core/agents/tools/retrievers/clinical_trials_retriever.py

import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode
import logging
from datetime import datetime

# Cookie警告修复
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Invalid attribute.*")
warnings.filterwarnings("ignore", message=".*Can not load response cookies.*")
logging.getLogger('aiohttp').setLevel(logging.ERROR)


logger = logging.getLogger(__name__)

class ClinicalTrialsRetriever:
    """专门的临床试验检索器 - 使用新版 ClinicalTrials.gov API v2"""
    
    def __init__(self):
        self.name = "ClinicalTrials Retriever"
        self.version = "2.0.0"
        self.base_url = "https://beta.clinicaltrials.gov/api/v2/studies"  # 新版API
        self.session = None
        self.sleep_sec = 0.3  # 请求间隔
        
        # 新版API字段映射
        self.field_mapping = {
            "nct_id": "nctId",
            "title": "briefTitle",
            "status": "overallStatus", 
            "phase": "phase",
            "sponsor": "leadSponsor",
            "condition": "conditions",
            "intervention": "interventions",
            "enrollment": "enrollment",
            "start_date": "startDate",
            "completion_date": "completionDate",
            "primary_outcome": "primaryOutcomes",
            "study_type": "studyType",
            "locations": "locations",
            "brief_summary": "briefSummary",
            "detailed_description": "detailedDescription"
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_by_gene(self, gene: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        根据基因名称搜索临床试验 - 使用新版API v2
        
        Args:
            gene: 基因名称 (如 "PCSK9", "EGFR")
            **kwargs: 额外过滤条件
                - condition: 适应症
                - phase: 试验阶段
                - status: 试验状态
                - page_size: 每页结果数 (默认20)
                - max_pages: 最大页数 (默认3)
        
        Returns:
            List[Dict]: 临床试验列表
        """
        
        # 处理参数冲突
        if gene is None and 'gene' in kwargs:
            gene = kwargs.pop('gene')
        elif 'gene' in kwargs:
            kwargs.pop('gene')
            
        if not gene:
            logger.warning("基因名称为空")
            return []
        
        if not self.session:
            self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
        
        try:
            # 使用新版API搜索
            trials = await self._search_trials_v2(gene, **kwargs)
            
            logger.info(f"为基因 {gene} 找到 {len(trials)} 个临床试验")
            return trials
            
        except Exception as e:
            logger.error(f"搜索基因 {gene} 的临床试验时出错: {str(e)}")
            return []
    
    async def _search_trials_v2(self, term: str, **kwargs) -> List[Dict[str, Any]]:
        """使用新版API v2搜索试验"""
        
        page_size = kwargs.get('page_size', 20)
        max_pages = kwargs.get('max_pages', 3)
        
        all_studies = []
        page_token = None
        
        for page in range(max_pages):
            params = {
                "query.titles": term,  # 在标题中搜索
                "pageSize": page_size,
            }
            
            # 添加额外的搜索条件
            if kwargs.get('condition'):
                params["query.cond"] = kwargs['condition']
            
            if kwargs.get('phase'):
                params["filter.phase"] = kwargs['phase']
                
            if kwargs.get('status'):
                params["filter.overallStatus"] = kwargs['status']
            
            if page_token:
                params["pageToken"] = page_token
            
            try:
                async with self.session.get(self.base_url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        studies = data.get("studies", [])
                        
                        if studies:
                            # 解析每个试验
                            parsed_studies = [self._parse_study_v2(study) for study in studies]
                            all_studies.extend(parsed_studies)
                        
                        # 获取下一页token
                        page_token = data.get("nextPageToken")
                        if not page_token:
                            break
                            
                    else:
                        logger.error(f"API请求失败: HTTP {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"第{page+1}页请求失败: {str(e)}")
                break
            
            # 请求间隔
            if page < max_pages - 1:  # 最后一页不需要等待
                await asyncio.sleep(self.sleep_sec)
        
        return all_studies
    
    def _parse_study_v2(self, study: Dict) -> Dict[str, Any]:
        """解析新版API v2的试验数据"""
        
        try:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            desc = protocol.get("descriptionModule", {})
            design = protocol.get("designModule", {})
            status = protocol.get("statusModule", {})
            sponsor = protocol.get("sponsorCollaboratorsModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            contacts = protocol.get("contactsLocationsModule", {})
            
            # 提取基本信息
            parsed = {
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status.get("overallStatus", "Unknown"),
                "phase": self._extract_phase_v2(design),
                "lead_sponsor": self._extract_sponsor_v2(sponsor),
                "condition": self._extract_conditions_v2(protocol),
                "interventions": self._extract_interventions_v2(protocol),
                "enrollment": self._extract_enrollment_v2(design),
                "start_date": self._extract_start_date_v2(status),
                "completion_date": self._extract_completion_date_v2(status),
                "outcomes": self._extract_outcomes_v2(protocol),
                "study_design": design.get("studyType", ""),
                "locations": self._extract_locations_v2(contacts),
                "brief_summary": self._extract_brief_summary_v2(desc),
                "detailed_description": self._extract_detailed_description_v2(desc)
            }
            
            return parsed
            
        except Exception as e:
            logger.error(f"解析试验数据时出错: {str(e)}")
            return {"nct_id": "", "title": "Parse Error", "status": "Unknown"}
    
    def _extract_phase_v2(self, design: Dict) -> str:
        """提取试验阶段 - 基于真实API结构"""
        # 真实API中没有phases字段，需要从其他地方推断
        study_type = design.get("studyType", "")
        design_info = design.get("designInfo", {})
        
        # 方法1：从studyType推断
        if study_type:
            study_type_lower = study_type.lower()
            if "interventional" in study_type_lower:
                # 进一步从设计信息推断阶段
                allocation = design_info.get("allocation", "").lower()
                intervention_model = design_info.get("interventionModel", "").lower()
                masking_info = design_info.get("maskingInfo", {})
                masking = masking_info.get("masking", "").lower() if masking_info else ""
                
                # 基于设计复杂度推断阶段
                if "randomized" in allocation and "parallel" in intervention_model:
                    if "quadruple" in masking or "triple" in masking:
                        return "Phase III"  # 通常大规模随机对照试验
                    elif "double" in masking:
                        return "Phase II"   # 通常中等规模试验
                    else:
                        return "Phase I/II" # 早期试验
                else:
                    return "Phase I"        # 非随机通常是早期
            
            elif "observational" in study_type_lower:
                return "Observational"
            elif "expanded access" in study_type_lower:
                return "Expanded Access"
        
        # 方法2：从入组人数推断（粗略估计）
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment_count = enrollment_info.get("count", 0)
        
        if enrollment_count > 0:
            if enrollment_count > 1000:
                return "Phase III"
            elif enrollment_count > 100:
                return "Phase II"
            elif enrollment_count > 20:
                return "Phase I/II"
            else:
                return "Phase I"
        
        return "Not Specified"
    
    def _extract_sponsor_v2(self, sponsor_module: Dict) -> str:
        """提取主要发起方 - 基于真实API结构"""
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name", "Unknown")
        sponsor_class = lead_sponsor.get("class", "")
        
        # 如果有sponsor class信息，加上分类
        if sponsor_class and sponsor_class != "Unknown":
            return f"{sponsor_name} ({sponsor_class})"
        
        return sponsor_name
    
    def _extract_conditions_v2(self, protocol: Dict) -> str:
        """提取适应症"""
        conditions_module = protocol.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])
        return "; ".join(conditions) if conditions else "Unknown"
    
    def _extract_interventions_v2(self, protocol: Dict) -> List[Dict]:
        """提取干预措施"""
        arms_module = protocol.get("armsInterventionsModule", {})
        interventions = arms_module.get("interventions", [])
        
        result = []
        for intervention in interventions:
            result.append({
                "name": intervention.get("name", ""),
                "type": intervention.get("type", ""),
                "description": intervention.get("description", "")
            })
        
        return result
    
    def _extract_enrollment_v2(self, design: Dict) -> Dict:
        """提取入组人数 - 基于真实API结构"""
        enrollment_info = design.get("enrollmentInfo", {})
        
        count = enrollment_info.get("count", 0)
        enrollment_type = enrollment_info.get("type", "Unknown")
        
        # 确保count是数字
        if isinstance(count, str):
            try:
                count = int(count)
            except ValueError:
                count = 0
        
        return {
            "count": count,
            "type": enrollment_type
        }
    
    def _extract_start_date_v2(self, status: Dict) -> str:
        """提取开始日期"""
        start_date = status.get("startDateStruct", {})
        return start_date.get("date", "")
    
    def _extract_completion_date_v2(self, status: Dict) -> str:
        """提取完成日期"""
        completion_date = status.get("completionDateStruct", {})
        return completion_date.get("date", "")
    
    def _extract_outcomes_v2(self, protocol: Dict) -> List[Dict]:
        """提取主要终点"""
        outcomes_module = protocol.get("outcomesModule", {})
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])
        
        result = []
        for outcome in primary_outcomes:
            result.append({
                "type": "Primary",
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", "")
            })
        
        return result
    
    def _extract_locations_v2(self, contacts: Dict) -> List[str]:
        """提取试验地点 - 基于真实API结构"""
        # 真实API中contactsLocationsModule只有centralContacts和overallOfficials
        # 需要从其他地方获取地点信息，或者改为显示联系信息
        
        central_contacts = contacts.get("centralContacts", [])
        overall_officials = contacts.get("overallOfficials", [])
        
        locations = []
        
        # 从联系人信息推断地点（如果有的话）
        for contact in central_contacts:
            # 通常联系人信息不包含具体地点
            name = contact.get("name", "")
            if name:
                locations.append(f"Contact: {name}")
        
        # 从研究负责人推断
        for official in overall_officials:
            name = official.get("name", "")
            affiliation = official.get("affiliation", "")
            if affiliation:
                locations.append(affiliation)
            elif name:
                locations.append(f"PI: {name}")
        
        # 如果没有找到地点信息，返回通用描述
        if not locations:
            return ["Location data not available in API"]
        
        return locations[:5]  # 最多返回5个
    
    def _extract_brief_summary_v2(self, desc: Dict) -> str:
        """提取简要总结"""
        brief_summary = desc.get("briefSummary", "")
        if isinstance(brief_summary, dict):
            return brief_summary.get("textBlock", "")
        return brief_summary
    
    def _extract_detailed_description_v2(self, desc: Dict) -> str:
        """提取详细描述"""
        detailed_desc = desc.get("detailedDescription", "")
        if isinstance(detailed_desc, dict):
            return detailed_desc.get("textBlock", "")
        return detailed_desc
    
    async def search_by_condition(self, condition: str, **kwargs) -> List[Dict[str, Any]]:
        """根据疾病/适应症搜索临床试验"""
        
        if not self.session:
            self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
            
        try:
            kwargs['condition'] = condition
            trials = await self._search_trials_v2(condition, **kwargs)
            
            logger.info(f"为疾病 {condition} 找到 {len(trials)} 个临床试验")
            return trials
            
        except Exception as e:
            logger.error(f"搜索疾病 {condition} 的临床试验时出错: {str(e)}")
            return []
    
    async def search_by_sponsor(self, sponsor: str, **kwargs) -> List[Dict[str, Any]]:
        """根据发起方搜索临床试验"""
        
        if not self.session:
            self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
            
        try:
            # 新版API暂时不支持直接按发起方搜索，使用通用搜索
            kwargs['sponsor'] = sponsor
            trials = await self._search_trials_v2(sponsor, **kwargs)
            
            logger.info(f"为发起方 {sponsor} 找到 {len(trials)} 个临床试验")
            return trials
            
        except Exception as e:
            logger.error(f"搜索发起方 {sponsor} 的临床试验时出错: {str(e)}")
            return []
    
    async def get_trial_details(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """获取特定试验的详细信息"""
        
        if not self.session:
            self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
        
        url = f"{self.base_url}/{nct_id}"
        
        try:
            async with self.session.get(url, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_study_v2(data)
                    
                return None
                
        except Exception as e:
            logger.error(f"获取试验 {nct_id} 详情时出错: {str(e)}")
            return None
    
    async def search_advanced(self, query_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """高级搜索接口"""
        
        if not self.session:
            self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            headers={
                'User-Agent': 'EpigenicAI/2.0.0',
                'Accept': 'application/json'
            }
        )
            
        try:
            # 提取搜索词
            search_term = query_dict.get("gene", "") or query_dict.get("condition", "")
            
            if not search_term:
                logger.warning("高级搜索需要基因或疾病名称")
                return []
            
            trials = await self._search_trials_v2(search_term, **query_dict)
            
            logger.info(f"高级搜索找到 {len(trials)} 个临床试验")
            return trials
            
        except Exception as e:
            logger.error(f"高级搜索时出错: {str(e)}")
            return []
    
    def get_supported_fields(self) -> List[str]:
        """获取支持的字段列表"""
        return list(self.field_mapping.keys())
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        return {
            "name": self.name,
            "version": self.version,
            "base_url": self.base_url,
            "api_version": "v2",
            "supported_fields": self.get_supported_fields(),
            "rate_limit": "请遵守ClinicalTrials.gov API使用限制",
            "notes": "使用新版 ClinicalTrials.gov API v2"
        }

# 使用示例
async def example_usage():
    """使用示例"""
    
    async with ClinicalTrialsRetriever() as retriever:
        
        # 1. 基因搜索
        print("=== 基因搜索测试 ===")
        pcsk9_trials = await retriever.search_by_gene("PCSK9", page_size=10, max_pages=2)
        print(f"PCSK9试验数量: {len(pcsk9_trials)}")
        
        if pcsk9_trials:
            trial = pcsk9_trials[0]
            print(f"示例试验: {trial['nct_id']} - {trial['title']}")
        
        # 2. 疾病搜索
        print("\n=== 疾病搜索测试 ===")
        cancer_trials = await retriever.search_by_condition("cancer", page_size=5, max_pages=1)
        print(f"癌症试验数量: {len(cancer_trials)}")
        
        # 3. 高级搜索
        print("\n=== 高级搜索测试 ===")
        advanced_results = await retriever.search_advanced({
            "gene": "EGFR",
            "condition": "lung cancer",
            "page_size": 5,
            "max_pages": 1
        })
        print(f"高级搜索结果: {len(advanced_results)}")
        
        # 4. API信息
        print("\n=== API信息 ===")
        api_info = retriever.get_api_info()
        print(f"API版本: {api_info['api_version']}")
        print(f"基础URL: {api_info['base_url']}")

if __name__ == "__main__":
    asyncio.run(example_usage())
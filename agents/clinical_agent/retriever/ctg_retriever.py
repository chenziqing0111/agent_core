import requests
import time
from typing import List, Dict, Optional

BETA_ENDPOINT = "https://beta.clinicaltrials.gov/api/v2/studies"


def get_trials_basic(term: str, *, page_size: int = 20, max_pages: int = 3, sleep_sec: float = 0.3) -> List[Dict]:
    """Return trial objects containing title + briefSummary + detailedDescription."""
    all_studies: List[Dict] = []
    token: Optional[str] = None

    for _ in range(max_pages):
        params = {
            "query.titles": term,
            "pageSize": page_size,
        }
        if token:
            params["pageToken"] = token

        resp = requests.get(BETA_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        all_studies.extend(data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(sleep_sec)

    return all_studies


def format_trials_brief(trials: List[Dict]) -> str:
    """Format trials for LLM summary and selection."""
    entries = []
    for trial in trials:
        ident = trial["protocolSection"]["identificationModule"]
        desc = trial["protocolSection"].get("descriptionModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        brief_data = desc.get("briefSummary", "")
        brief = brief_data.get("textBlock", "") if isinstance(brief_data, dict) else brief_data

        entries.append(f"[{nct_id}] {title}\n{brief}")

    return "\n\n".join(entries)


def format_detailed_analysis(trials: List[Dict], top_nct_ids: List[str]) -> str:
    """Format selected trials for second-round deep analysis."""
    blocks = []
    for trial in trials:
        ident = trial["protocolSection"]["identificationModule"]
        desc = trial["protocolSection"].get("descriptionModule", {})

        nct_id = ident.get("nctId", "")
        if nct_id not in top_nct_ids:
            continue

        title = ident.get("briefTitle", "")
        detail_data = desc.get("detailedDescription", "")
        detail = detail_data.get("textBlock", "") if isinstance(detail_data, dict) else detail_data

        blocks.append(f"### {title} ({nct_id})\n\n{detail}")

    return "\n\n".join(blocks)

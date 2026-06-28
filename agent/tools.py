"""
Agent Tools — 4 tools, 3 different data sources

Tool 1: search_policy_coverage      → Azure AI Search (coverage/costs)
Tool 2: search_prior_auth_criteria  → Azure AI Search (approval criteria)
Tool 3: check_drug_interaction_fda  → FDA openFDA API (live external)
Tool 4: get_cms_coverage_data       → CMS data API (live external)

GPT-4o reads the tool descriptions and decides which one to call.
Each tool returns different type of information from different source.
"""

import os
import requests
from datetime import date
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# --- Clients ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX", "healthcare-docs"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)


def get_embedding(text: str) -> list[float]:
    """Convert text to vector for semantic search."""
    response = openai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=text
    )
    return response.data[0].embedding


# ============================================================
# TOOL 1: Policy Coverage Search
# Source: Azure AI Search — policy_general.txt focus
# Purpose: Answer "is X covered?" and "how much does X cost?"
# ============================================================

def search_policy_coverage(query: str) -> str:
    """
    Search insurance policy documents for coverage information.

    Answers questions about:
    - What services are covered
    - Copay and coinsurance amounts
    - Deductible and out-of-pocket limits
    - In-network vs out-of-network benefits
    - Annual visit limits

    Uses hybrid search (keyword + vector) against Azure AI Search.
    Prioritizes policy_general.txt which contains benefits schedule.

    Different from Tool 2 which focuses on prior auth criteria.
    This tool answers COVERAGE questions, not APPROVAL questions.
    """
    query_embedding = get_embedding(query)

    results = search_client.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=3,
                fields="content_vector"
            )
        ],
        # Filter to prioritize general policy and formulary
        # These contain coverage details, costs, and limits
        filter="source_file eq 'policy_general.txt' "
               "or source_file eq 'drug_formulary.txt'",
        select=["content", "source_file", "chunk_number"],
        top=3
    )

    chunks = []
    for i, result in enumerate(results):
        chunks.append(
            f"[Source: {result['source_file']}, "
            f"section {result['chunk_number']}]\n"
            f"{result['content']}"
        )

    if not chunks:
        # No filter match — search all documents
        results = search_client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=3,
                    fields="content_vector"
                )
            ],
            select=["content", "source_file", "chunk_number"],
            top=3
        )
        for i, result in enumerate(results):
            chunks.append(
                f"[Source: {result['source_file']}, "
                f"section {result['chunk_number']}]\n"
                f"{result['content']}"
            )

    if not chunks:
        return "No coverage information found for this query."

    return (
        f"COVERAGE INFORMATION\n"
        f"Query: {query}\n"
        f"Source: Azure AI Search — Policy Documents\n"
        f"{'='*50}\n\n"
        + "\n\n---\n\n".join(chunks)
    )


# ============================================================
# TOOL 2: Prior Auth Criteria Search
# Source: Azure AI Search — policy_procedures.txt focus
# Purpose: Answer "what do I need to submit for approval?"
# ============================================================

def search_prior_auth_criteria(procedure: str) -> str:
    """
    Search for prior authorization and medical necessity criteria.

    Answers questions about:
    - What documentation is required for approval
    - Medical necessity criteria that must be met
    - Step therapy requirements (try X before Y)
    - Clinical criteria checklist for specific procedures
    - Appeal process and timelines

    Targets policy_procedures.txt which contains
    the medical necessity and prior auth sections.

    Different from Tool 1 which answers coverage/cost questions.
    This tool answers APPROVAL CRITERIA questions — what must be
    documented and proven before the insurer will approve.
    """
    # Build a targeted query focused on prior auth criteria
    targeted_query = (
        f"prior authorization medical necessity criteria "
        f"documentation requirements {procedure}"
    )
    query_embedding = get_embedding(targeted_query)

    results = search_client.search(
        search_text=targeted_query,
        vector_queries=[
            VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=3,
                fields="content_vector"
            )
        ],
        # Target the procedures document specifically
        filter="source_file eq 'policy_procedures.txt'",
        select=["content", "source_file", "chunk_number"],
        top=3
    )

    chunks = []
    for i, result in enumerate(results):
        chunks.append(
            f"[Source: {result['source_file']}, "
            f"section {result['chunk_number']}]\n"
            f"{result['content']}"
        )

    if not chunks:
        # Fallback — search all docs
        results = search_client.search(
            search_text=targeted_query,
            vector_queries=[
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=3,
                    fields="content_vector"
                )
            ],
            select=["content", "source_file", "chunk_number"],
            top=3
        )
        for i, result in enumerate(results):
            chunks.append(
                f"[Source: {result['source_file']}, "
                f"section {result['chunk_number']}]\n"
                f"{result['content']}"
            )

    if not chunks:
        return (
            f"No prior authorization criteria found "
            f"for {procedure}."
        )

    return (
        f"PRIOR AUTHORIZATION CRITERIA\n"
        f"Procedure: {procedure}\n"
        f"Source: Azure AI Search — Procedures Policy\n"
        f"{'='*50}\n\n"
        + "\n\n---\n\n".join(chunks)
    )


# ============================================================
# TOOL 3: FDA Drug Interaction
# Source: api.fda.gov — LIVE EXTERNAL API
# Purpose: Real FDA drug label interaction data
# ============================================================

def check_drug_interaction_fda(drug1: str, drug2: str) -> str:
    """
    Fetch real drug interaction data from FDA openFDA API.

    Data source: Official FDA drug prescribing labels.
    Contains the drug_interactions section that pharmacists
    and doctors use when prescribing. Updated by FDA directly.

    This is a LIVE external API call — not from our documents.
    Returns authoritative FDA data, not synthetic policy text.

    Includes timeout handling and fallback to local search
    if FDA API is unavailable.
    """
    BASE_URL = "https://api.fda.gov/drug/label.json"
    fda_results = {}

    for drug in [drug1, drug2]:
        try:
            response = requests.get(
                BASE_URL,
                params={
                         "search": f"openfda.generic_name:{drug}",
                         "limit": 1
                      },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    label = data["results"][0]
                    interactions = label.get("drug_interactions", [])
                    fda_results[drug] = (
                        interactions[0] if interactions else None
                    )
                else:
                    fda_results[drug] = None
            else:
                fda_results[drug] = None

        except requests.Timeout:
            return (
                f"FDA API timeout. "
                f"Falling back to local knowledge base.\n\n"
                + search_policy_coverage(
                    f"drug interaction {drug1} {drug2}"
                )
            )
        except requests.RequestException:
            fda_results[drug] = None

    # Build output
    output = [
        f"FDA DRUG INTERACTION REPORT",
        f"Drugs: {drug1.upper()} + {drug2.upper()}",
        f"Source: openFDA Official Drug Labels (api.fda.gov)",
        f"Retrieved: {date.today()}",
        "=" * 60
    ]

    # Drug1 label
    if fda_results.get(drug1):
        text = fda_results[drug1]
        output.append(f"\n{drug1.upper()} — Interactions Section:")

        # Find sentences mentioning drug2
        drug2_lower = drug2.lower()
        if drug2_lower in text.lower():
            sentences = text.split(".")
            relevant = [
                s.strip() for s in sentences
                if drug2_lower in s.lower()
                and len(s.strip()) > 20
            ]
            if relevant:
                output.append(
                    f"Mentions of {drug2}: "
                    + ". ".join(relevant[:3]) + "."
                )

        output.append(f"\nFull section (first 500 chars):")
        output.append(text[:500] + "...")
    else:
        output.append(
            f"\n{drug1.upper()}: No FDA label found. "
            f"Try exact generic name."
        )

    # Drug2 label
    if fda_results.get(drug2):
        text = fda_results[drug2]
        output.append(f"\n{drug2.upper()} — Interactions Section:")

        drug1_lower = drug1.lower()
        if drug1_lower in text.lower():
            sentences = text.split(".")
            relevant = [
                s.strip() for s in sentences
                if drug1_lower in s.lower()
                and len(s.strip()) > 20
            ]
            if relevant:
                output.append(
                    f"Mentions of {drug1}: "
                    + ". ".join(relevant[:3]) + "."
                )

        output.append(f"\nFull section (first 500 chars):")
        output.append(text[:500] + "...")
    else:
        output.append(
            f"\n{drug2.upper()}: No FDA label found."
        )

    output.append(
        "\n⚠ DISCLAIMER: openFDA data is for informational "
        "purposes only. Always verify with a licensed pharmacist "
        "or physician before clinical decisions."
    )

    return "\n".join(output)


# ============================================================
# TOOL 4: CMS Medicare Coverage Data
# Source: data.cms.gov — LIVE EXTERNAL API
# Purpose: Real Medicare coverage and ACO data from CMS
# ============================================================

def get_cms_coverage_data(query: str) -> str:
    """
    Fetch real Medicare coverage data from CMS data API.

    Data source: Centers for Medicare & Medicaid Services (CMS)
    data.cms.gov — official government healthcare data.

    Returns real ACO (Accountable Care Organization) data,
    Medicare coverage policies, and payment information.

    This is completely different from our synthetic documents —
    it is real US government Medicare data updated by CMS.

    Use cases:
    - Medicare coverage questions
    - ACO and value-based care questions
    - Medicare payment and reimbursement questions
    """
    CMS_BASE = "https://data.cms.gov/data-api/v1/dataset"

    # ACO Participant dataset — real Medicare ACO data
    ACO_DATASET = (
        "9767cb68-8ea9-4f0b-8179-9431abc89f11"
    )

    output = [
        f"CMS MEDICARE DATA",
        f"Query: {query}",
        f"Source: data.cms.gov — Official CMS Database",
        f"Retrieved: {date.today()}",
        "=" * 60
    ]

    try:
        # Search CMS ACO dataset
        response = requests.get(
            f"{CMS_BASE}/{ACO_DATASET}/data",
            params={
                "size": 3,
                "keyword": query
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                output.append(
                    f"\nFound {len(data)} CMS records:\n"
                )
                for i, record in enumerate(data[:3]):
                    output.append(f"Record {i+1}:")

                    # Extract most relevant fields
                    relevant_fields = [
                        "ACO_Name", "ACO_Service_Area",
                        "ACO_Public_Name", "Par_LBN",
                        "Agreement_Period_Num",
                        "Current_Start_Date",
                        "ENHANCED_Track", "High_Revenue_ACO",
                        "Low_Revenue_ACO"
                    ]

                    for field in relevant_fields:
                        if field in record and record[field]:
                            # Make field name readable
                            readable = field.replace("_", " ")
                            output.append(
                                f"  {readable}: {record[field]}"
                            )
                    output.append("")

            else:
                output.append(
                    "\nNo CMS records found for this query. "
                    "Try broader search terms like "
                    "'Medicare' or state abbreviation."
                )

        elif response.status_code == 404:
            output.append("\nCMS dataset not available.")
        else:
            output.append(
                f"\nCMS API returned status {response.status_code}"
            )

    except requests.Timeout:
        output.append(
            "\nCMS API timeout. "
            "Try again or check data.cms.gov directly."
        )
    except requests.RequestException as e:
        output.append(f"\nCMS API error: {str(e)}")

    output.append(
        "\nFor full CMS data visit: https://data.cms.gov"
    )

    return "\n".join(output)


# ============================================================
# Tool definitions — JSON schema GPT-4o reads
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_coverage",
            "description": (
                "Search insurance policy documents to answer "
                "coverage questions: what is covered, copay amounts, "
                "deductibles, visit limits, in/out of network costs. "
                "Use for: 'Is X covered?', 'How much is my copay?', "
                "'What is my deductible?', 'How many PT visits?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Coverage question to search. "
                            "Example: 'physical therapy copay coverage'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_prior_auth_criteria",
            "description": (
                "Search for prior authorization requirements and "
                "medical necessity criteria for a specific procedure. "
                "Use for: 'What do I need to submit for prior auth?', "
                "'What are the approval criteria for X?', "
                "'What documentation is needed for knee replacement?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure": {
                        "type": "string",
                        "description": (
                            "Procedure name or CPT code. "
                            "Example: 'total knee replacement' "
                            "or 'physical therapy'"
                        )
                    }
                },
                "required": ["procedure"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_interaction_fda",
            "description": (
                "Get REAL drug interaction data from FDA official "
                "drug prescribing labels via openFDA API. "
                "Use for any drug interaction or medication safety "
                "question. Returns authoritative FDA data, "
                "not synthetic documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug1": {
                        "type": "string",
                        "description": "First drug generic name"
                    },
                    "drug2": {
                        "type": "string",
                        "description": "Second drug generic name"
                    }
                },
                "required": ["drug1", "drug2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cms_coverage_data",
            "description": (
                "Get real Medicare and Medicaid coverage data "
                "from CMS (Centers for Medicare & Medicaid Services) "
                "official database. Use for Medicare coverage "
                "questions, ACO information, and government "
                "healthcare program questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Medicare/CMS search query. "
                            "Example: 'Medicare knee replacement' "
                            "or 'ACO Texas'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Dispatch tool calls from GPT-4o to actual functions.
    """
    if tool_name == "search_policy_coverage":
        return search_policy_coverage(**tool_args)
    elif tool_name == "search_prior_auth_criteria":
        return search_prior_auth_criteria(**tool_args)
    elif tool_name == "check_drug_interaction_fda":
        return check_drug_interaction_fda(**tool_args)
    elif tool_name == "get_cms_coverage_data":
        return get_cms_coverage_data(**tool_args)
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================
# Test all 4 tools
# ============================================================

if __name__ == "__main__":
    print("=== TOOLS TEST — 4 different data sources ===\n")

    print("TOOL 1: Policy coverage (Azure AI Search)")
    print("-" * 50)
    result = search_policy_coverage(
        "physical therapy copay and visit limit"
    )
    print(result[:400])
    print()

    print("TOOL 2: Prior auth criteria (Azure AI Search filtered)")
    print("-" * 50)
    result = search_prior_auth_criteria("total knee replacement")
    print(result[:400])
    print()

    print("TOOL 3: FDA drug interaction (live api.fda.gov)")
    print("-" * 50)
    result = check_drug_interaction_fda("metformin", "lisinopril")
    print(result[:600])
    print()

    print("TOOL 4: CMS Medicare data (live data.cms.gov)")
    print("-" * 50)
    result = get_cms_coverage_data("Medicare Texas")
    print(result[:400])
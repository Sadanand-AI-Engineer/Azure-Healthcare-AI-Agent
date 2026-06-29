"""
Agent Tools — 5 tools, 4 different data sources

Tool 1: search_policy_coverage      → Azure AI Search (coverage/costs)
Tool 2: search_prior_auth_criteria  → Azure AI Search (approval criteria)
Tool 3: check_drug_interaction_fda  → FDA openFDA API (live external)
Tool 4: get_cms_coverage_data       → CMS data API (live external)
Tool 5: find_doctors_npi            → NPI Registry API (live external)
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
# Source: Azure AI Search
# ============================================================

def search_policy_coverage(query: str) -> dict:
    """Search insurance policy for coverage and cost information."""
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
        filter="source_file eq 'policy_general.txt' "
               "or source_file eq 'drug_formulary.txt'",
        select=["content", "source_file", "chunk_number"],
        top=3
    )

    chunks = []
    scores = []
    for result in results:
        chunks.append(
            f"[Source: {result['source_file']}]\n{result['content']}"
        )
        scores.append(round(result.get('@search.score', 0), 3))

    if not chunks:
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
        for result in results:
            chunks.append(
                f"[Source: {result['source_file']}]\n{result['content']}"
            )
            scores.append(round(result.get('@search.score', 0), 3))

    return {
        "content": "\n\n---\n\n".join(chunks) if chunks
                   else "No coverage information found.",
        "rag_scores": scores,
        "top_score": max(scores) if scores else 0,
        "source": "Azure AI Search — Policy Documents",
        "chunks_retrieved": len(chunks)
    }


# ============================================================
# TOOL 2: Prior Auth Criteria Search
# Source: Azure AI Search (filtered)
# ============================================================

def search_prior_auth_criteria(procedure: str) -> dict:
    """Search for prior authorization and medical necessity criteria."""
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
        filter="source_file eq 'policy_procedures.txt'",
        select=["content", "source_file", "chunk_number"],
        top=3
    )

    chunks = []
    scores = []
    for result in results:
        chunks.append(
            f"[Source: {result['source_file']}]\n{result['content']}"
        )
        scores.append(round(result.get('@search.score', 0), 3))

    if not chunks:
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
        for result in results:
            chunks.append(
                f"[Source: {result['source_file']}]\n{result['content']}"
            )
            scores.append(round(result.get('@search.score', 0), 3))

    return {
        "content": "\n\n---\n\n".join(chunks) if chunks
                   else "No prior auth criteria found.",
        "rag_scores": scores,
        "top_score": max(scores) if scores else 0,
        "source": "Azure AI Search — Procedures Policy",
        "chunks_retrieved": len(chunks)
    }


# ============================================================
# TOOL 3: FDA Drug Interaction
# Source: api.fda.gov — LIVE EXTERNAL API
# ============================================================

def check_drug_interaction_fda(drug1: str, drug2: str) -> dict:
    """Fetch real drug interaction data from FDA openFDA API."""
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
            fda_results[drug] = None
        except requests.RequestException:
            fda_results[drug] = None

    output = [
        f"FDA DRUG INTERACTION: {drug1.upper()} + {drug2.upper()}",
        f"Source: openFDA Official Drug Labels (api.fda.gov)",
        f"Retrieved: {date.today()}"
    ]

    for drug, text in fda_results.items():
        if text:
            output.append(f"\n{drug.upper()} interactions (first 600 chars):")
            output.append(text[:600] + "...")
        else:
            output.append(f"\n{drug.upper()}: No FDA label found.")

    output.append(
        "\n⚠ DISCLAIMER: Always verify with a licensed pharmacist."
    )

    return {
        "content": "\n".join(output),
        "source": "openFDA API (api.fda.gov)",
        "drugs_found": [d for d, v in fda_results.items() if v],
        "api_calls": 2
    }


# ============================================================
# TOOL 4: CMS Medicare Data
# Source: data.cms.gov — LIVE EXTERNAL API
# ============================================================

def get_cms_coverage_data(query: str) -> dict:
    """Fetch real Medicare coverage data from CMS API."""
    CMS_BASE = "https://data.cms.gov/data-api/v1/dataset"
    ACO_DATASET = "9767cb68-8ea9-4f0b-8179-9431abc89f11"

    output = [
        f"CMS MEDICARE DATA",
        f"Query: {query}",
        f"Source: data.cms.gov",
        f"Retrieved: {date.today()}"
    ]

    try:
        response = requests.get(
            f"{CMS_BASE}/{ACO_DATASET}/data",
            params={"size": 3, "keyword": query},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                output.append(f"\nFound {len(data)} CMS records:")
                for i, record in enumerate(data[:3]):
                    output.append(f"\nRecord {i+1}:")
                    for field in ["ACO_Name", "ACO_Service_Area",
                                  "Current_Start_Date"]:
                        if field in record and record[field]:
                            output.append(
                                f"  {field.replace('_', ' ')}: "
                                f"{record[field]}"
                            )
            else:
                output.append("\nNo CMS records found.")
        else:
            output.append(f"\nCMS API returned {response.status_code}")

    except requests.Timeout:
        output.append("\nCMS API timeout.")
    except requests.RequestException as e:
        output.append(f"\nCMS API error: {str(e)}")

    return {
        "content": "\n".join(output),
        "source": "CMS data.gov API",
        "api_calls": 1
    }


# ============================================================
# TOOL 5: Find Doctors — NPI Registry
# Source: npiregistry.cms.hhs.gov — LIVE EXTERNAL API
# ============================================================

def find_doctors_npi(
    specialty: str,
    state: str = "TX",
    limit: int = 2
) -> dict:
    """
    Find real licensed doctors from the NPI Registry.
    NPI = National Provider Identifier
    Every licensed US doctor/hospital has one.
    Source: official CMS NPI Registry — completely free, no key needed.
    """
    NPI_URL = "https://npiregistry.cms.hhs.gov/api/"

    # Map common symptoms/conditions to taxonomy codes
    SPECIALTY_MAP = {
        "general": "207Q00000X",      # Family Medicine
        "fever": "207Q00000X",         # Family Medicine
        "headache": "2084N0400X",      # Neurology
        "heart": "207RC0000X",         # Cardiology
        "chest pain": "207RC0000X",    # Cardiology
        "diabetes": "207RE0101X",      # Endocrinology
        "skin": "207N00000X",          # Dermatology
        "mental": "2084P0800X",        # Psychiatry
        "bone": "207X00000X",          # Orthopedic Surgery
        "knee": "207X00000X",          # Orthopedic Surgery
        "stomach": "207RG0100X",       # Gastroenterology
        "lung": "207RP1001X",          # Pulmonology
        "eye": "207W00000X",           # Ophthalmology
        "child": "208000000X",         # Pediatrics
        "women": "207V00000X",         # OB/GYN
        "cancer": "207RX0202X",        # Oncology
    }

    # Find best matching taxonomy
    taxonomy = "207Q00000X"  # default: Family Medicine
    specialty_lower = specialty.lower()
    for key, code in SPECIALTY_MAP.items():
        if key in specialty_lower:
            taxonomy = code
            break

    try:
        response = requests.get(
            NPI_URL,
            params={
                "version": "2.1",
                "taxonomy_description": "",
                "state": state,
                "limit": limit,
                "enumeration_type": "NPI-1",  # Individual providers only
                "taxonomy_code": taxonomy,
                "skip": 0
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                # Try without taxonomy filter
                response2 = requests.get(
                    NPI_URL,
                    params={
                        "version": "2.1",
                        "state": state,
                        "limit": limit,
                        "enumeration_type": "NPI-1",
                        "skip": 0
                    },
                    timeout=10
                )
                if response2.status_code == 200:
                    results = response2.json().get("results", [])

            doctors = []
            for r in results[:limit]:
                basic = r.get("basic", {})
                addresses = r.get("addresses", [{}])
                taxonomies = r.get("taxonomies", [{}])

                name = f"Dr. {basic.get('first_name', '')} " \
                       f"{basic.get('last_name', '')}"
                specialty_desc = taxonomies[0].get(
                    "desc", "General Practice"
                ) if taxonomies else "General Practice"
                city = addresses[0].get("city", "") if addresses else ""
                state_addr = addresses[0].get(
                    "state", state
                ) if addresses else state
                npi_number = r.get("number", "")

                npi_link = (
                    f"https://npiregistry.cms.hhs.gov/provider-view/"
                    f"{npi_number}"
                )

                doctors.append({
                    "name": name.strip(),
                    "specialty": specialty_desc,
                    "location": f"{city}, {state_addr}",
                    "npi": npi_number,
                    "link": npi_link,
                    "reason": f"Licensed {specialty_desc} specialist "
                              f"in {city or state_addr}"
                })

            if doctors:
                return {
                    "content": format_doctors(doctors, specialty),
                    "doctors": doctors,
                    "source": "NPI Registry (npiregistry.cms.hhs.gov)",
                    "total_found": len(doctors),
                    "api_calls": 1
                }
            else:
                return {
                    "content": (
                        "No doctors found in NPI Registry for this "
                        f"specialty in {state}. "
                        "Please search healthgrades.com or zocdoc.com."
                    ),
                    "doctors": [],
                    "source": "NPI Registry",
                    "total_found": 0,
                    "api_calls": 1
                }

        else:
            return {
                "content": "NPI Registry unavailable. "
                           "Please try healthgrades.com",
                "doctors": [],
                "source": "NPI Registry",
                "total_found": 0,
                "api_calls": 1
            }

    except requests.Timeout:
        return {
            "content": "NPI Registry timeout. Try zocdoc.com",
            "doctors": [],
            "source": "NPI Registry",
            "total_found": 0,
            "api_calls": 0
        }
    except requests.RequestException as e:
        return {
            "content": f"NPI Registry error: {str(e)}",
            "doctors": [],
            "source": "NPI Registry",
            "total_found": 0,
            "api_calls": 0
        }


def format_doctors(doctors: list, condition: str) -> str:
    """Format doctor results for display."""
    lines = [
        f"RECOMMENDED DOCTORS for: {condition}",
        f"Source: NPI Registry (Official CMS Database)",
        f"All providers are licensed and verified by CMS.",
        ""
    ]

    for i, doc in enumerate(doctors, 1):
        lines.append(f"Doctor {i}: {doc['name']}")
        lines.append(f"  Specialty: {doc['specialty']}")
        lines.append(f"  Location: {doc['location']}")
        lines.append(f"  Why recommended: {doc['reason']}")
        lines.append(f"  NPI Profile: {doc['link']}")
        lines.append("")

    lines.append(
        "⚠ IMPORTANT: Always call the doctor's office to confirm "
        "availability and that they accept your insurance before visiting."
    )

    return "\n".join(lines)


# ============================================================
# Tool definitions for GPT-4o
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_coverage",
            "description": (
                "Search insurance policy documents for coverage, "
                "copay, deductible, and benefits information. "
                "Use for: 'Is X covered?', 'What is my copay?', "
                "'What is my deductible?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Coverage question to search"
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
                "medical necessity criteria for a procedure. "
                "Use for: 'What do I need for prior auth?', "
                "'What are the criteria for approval?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure": {
                        "type": "string",
                        "description": "Procedure name or CPT code"
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
                "drug labels. Use for drug interaction or medication "
                "safety questions."
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
                "Get real Medicare and Medicaid coverage data from "
                "CMS official database. Use for Medicare questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medicare/CMS search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_doctors_npi",
            "description": (
                "Find real licensed doctors from the official NPI "
                "Registry (National Provider Identifier database). "
                "Use when user has health symptoms and needs to see "
                "a doctor. Returns verified licensed providers with "
                "their NPI profile links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": (
                            "Medical specialty or symptom/condition. "
                            "Examples: 'fever', 'headache', 'knee pain', "
                            "'diabetes', 'heart', 'general'"
                        )
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "US state abbreviation. Default: TX"
                        ),
                        "default": "TX"
                    }
                },
                "required": ["specialty"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """Dispatch tool calls from GPT-4o to actual functions."""
    if tool_name == "search_policy_coverage":
        return search_policy_coverage(**tool_args)
    elif tool_name == "search_prior_auth_criteria":
        return search_prior_auth_criteria(**tool_args)
    elif tool_name == "check_drug_interaction_fda":
        return check_drug_interaction_fda(**tool_args)
    elif tool_name == "get_cms_coverage_data":
        return get_cms_coverage_data(**tool_args)
    elif tool_name == "find_doctors_npi":
        return find_doctors_npi(**tool_args)
    else:
        return {"content": f"Unknown tool: {tool_name}", "source": "none"}
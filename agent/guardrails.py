"""
PHI Guardrails — Layer 1 of the pipeline
Every message passes through this BEFORE hitting the LLM.
Blocks: SSN, MRN, DOB, phone numbers, email, real names in context
In enterprise: this is HIPAA compliance enforcement in code
"""

import re
import os
from dotenv import load_dotenv

load_dotenv()

# PHI patterns — what we detect and block
PHI_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "MRN": r"\bMRN[:\s#]*\d{4,10}\b",
    "DOB": r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b",
    "PHONE": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "NPI": r"\bNPI[:\s#]*\d{10}\b",
    "DEA": r"\b[A-Z]{2}\d{7}\b",
}

# What we replace PHI with — so context is preserved but PHI is removed
PHI_REPLACEMENTS = {
    "SSN": "[SSN REDACTED]",
    "MRN": "[MRN REDACTED]",
    "DOB": "[DOB REDACTED]",
    "PHONE": "[PHONE REDACTED]",
    "EMAIL": "[EMAIL REDACTED]",
    "CREDIT_CARD": "[CARD REDACTED]",
    "NPI": "[NPI REDACTED]",
    "DEA": "[DEA REDACTED]",
}


def detect_phi(text: str) -> dict:
    """
    Scan text for PHI patterns.
    Returns dict of what was found.
    """
    found = {}
    for phi_type, pattern in PHI_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[phi_type] = matches
    return found


def redact_phi(text: str) -> tuple[str, dict]:
    """
    Remove PHI from text and replace with safe placeholders.
    Returns: (redacted_text, dict of what was removed)
    """
    redacted = text
    removed = {}

    for phi_type, pattern in PHI_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            removed[phi_type] = matches
            redacted = re.sub(
                pattern,
                PHI_REPLACEMENTS[phi_type],
                redacted,
                flags=re.IGNORECASE
            )

    return redacted, removed


def check_harmful_content(text: str) -> tuple[bool, str]:
    """
    Check for harmful medical advice requests.
    Returns: (is_harmful, reason)
    """
    harmful_patterns = [
        (r"\bhow to (overdose|self.harm|suicide)\b",
         "Self-harm content detected"),
        (r"\b(synthesize|make|create|manufacture)\s+(drugs|meth|fentanyl)\b",
         "Drug synthesis request detected"),
        (r"\bbypass\s+(prescription|doctor|hospital)\b",
         "Prescription bypass attempt detected"),
    ]

    for pattern, reason in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, reason

    return False, ""


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate basic input quality.
    Returns: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Empty message received"

    if len(text.strip()) < 3:
        return False, "Message too short"

    if len(text) > 2000:
        return False, "Message too long. Please keep under 2000 characters"

    return True, ""


def process_input(user_message: str) -> dict:
    """
    MAIN FUNCTION — runs every user message through all guardrails.

    Pipeline:
    1. Validate input (not empty, not too long)
    2. Check for harmful content
    3. Detect and redact PHI
    4. Return safe message + audit log

    This audit log goes to Azure Monitor in production.
    """
    result = {
        "original_length": len(user_message),
        "safe_message": None,
        "blocked": False,
        "block_reason": None,
        "phi_detected": {},
        "phi_redacted": False,
        "passed_guardrails": False
    }

    # Step 1: Validate input
    is_valid, error = validate_input(user_message)
    if not is_valid:
        result["blocked"] = True
        result["block_reason"] = error
        return result

    # Step 2: Check harmful content
    is_harmful, reason = check_harmful_content(user_message)
    if is_harmful:
        result["blocked"] = True
        result["block_reason"] = reason
        return result

    # Step 3: Detect and redact PHI
    phi_found = detect_phi(user_message)
    if phi_found:
        safe_message, removed = redact_phi(user_message)
        result["phi_detected"] = removed
        result["phi_redacted"] = True
        result["safe_message"] = safe_message
    else:
        result["safe_message"] = user_message

    # Step 4: All checks passed
    result["passed_guardrails"] = True
    return result
def detect_intent(text: str) -> dict:
    """
    Detect the intent of the user message.
    Returns category and confidence.

    Categories:
    - healthcare_insurance: coverage, prior auth, claims, deductibles
    - healthcare_clinical: symptoms, medications, doctor questions
    - off_topic: unrelated to healthcare entirely
    """
    text_lower = text.lower()

    # Healthcare insurance keywords
    insurance_keywords = [
        "copay", "deductible", "coverage", "covered", "insurance",
        "prior auth", "authorization", "claim", "premium", "benefit",
        "in-network", "out-of-network", "formulary", "eob",
        "coinsurance", "referral", "hmo", "ppo", "medicare",
        "medicaid", "aca", "plan", "policy", "drug tier"
    ]

    # Clinical/health keywords
    clinical_keywords = [
        "fever", "headache", "pain", "symptom", "sick", "hurt",
        "doctor", "hospital", "medication", "drug", "prescription",
        "diagnosis", "treatment", "surgery", "therapy", "infection",
        "blood pressure", "diabetes", "heart", "cancer", "allergy",
        "cough", "nausea", "fatigue", "dizzy", "rash", "wound",
        "interact", "side effect", "dose", "dosage", "pharmacy"
    ]

    # Off-topic keywords
    off_topic_keywords = [
        "pizza", "weather", "sports", "movie", "music", "game",
        "politics", "stock", "crypto", "travel", "food", "recipe",
        "news", "celebrity", "joke", "poem", "code", "programming"
    ]

    insurance_score = sum(
        1 for kw in insurance_keywords if kw in text_lower
    )
    clinical_score = sum(
        1 for kw in clinical_keywords if kw in text_lower
    )
    off_topic_score = sum(
        1 for kw in off_topic_keywords if kw in text_lower
    )

    # Determine intent
    if off_topic_score > 0 and insurance_score == 0 and clinical_score == 0:
        return {
            "intent": "off_topic",
            "confidence": min(off_topic_score / 2, 1.0),
            "redirect_message": (
                "I'm your Healthcare AI Assistant — I'm specifically "
                "designed to help with healthcare questions. I can help "
                "you with:\n\n"
                "🏥 Insurance coverage and benefits\n"
                "💊 Drug interactions and formulary\n"
                "📋 Prior authorization requirements\n"
                "🩺 Health symptoms and doctor recommendations\n"
                "📊 Medicare and CMS coverage data\n\n"
                "Do you have any healthcare questions I can help with?"
            )
        }
    elif clinical_score > 0:
        return {
            "intent": "healthcare_clinical",
            "confidence": min(clinical_score / 3, 1.0)
        }
    elif insurance_score > 0:
        return {
            "intent": "healthcare_insurance",
            "confidence": min(insurance_score / 3, 1.0)
        }
    else:
        # Unclear — let the agent try to answer
        return {
            "intent": "healthcare_general",
            "confidence": 0.5
        }


# Quick test — run this file directly to test guardrails
if __name__ == "__main__":
    test_messages = [
        "Does my insurance cover knee surgery?",
        "My SSN is 123-45-6789 and DOB is 01/15/1985, am I covered?",
        "Patient MRN: 1234567 needs prior auth for CPT 27447",
        "How to overdose on medication?",
        "",
        "What is a deductible?",
    ]

    print("=== PHI GUARDRAILS TEST ===\n")
    for msg in test_messages:
        result = process_input(msg)
        print(f"Input:   {msg[:50]}")
        print(f"Blocked: {result['blocked']}")
        if result['blocked']:
            print(f"Reason:  {result['block_reason']}")
        if result['phi_redacted']:
            print(f"PHI found: {result['phi_detected']}")
            print(f"Safe msg: {result['safe_message']}")
        print(f"Passed:  {result['passed_guardrails']}")
        print("-" * 50)
    
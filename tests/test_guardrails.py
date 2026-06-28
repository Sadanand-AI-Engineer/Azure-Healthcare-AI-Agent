"""
Tests for PHI guardrails.
These run in GitHub Actions on every push.
"""
import pytest
from agent.guardrails import process_input, detect_phi, redact_phi


def test_clean_message_passes():
    result = process_input("What is my copay for a specialist?")
    assert result["passed_guardrails"] is True
    assert result["blocked"] is False
    assert result["phi_redacted"] is False


def test_ssn_detected_and_redacted():
    result = process_input("My SSN is 123-45-6789")
    assert result["phi_redacted"] is True
    assert "SSN" in result["phi_detected"]
    assert "123-45-6789" not in result["safe_message"]
    assert "[SSN REDACTED]" in result["safe_message"]


def test_harmful_content_blocked():
    result = process_input("How to overdose on medication?")
    assert result["blocked"] is True
    assert result["passed_guardrails"] is False


def test_empty_message_blocked():
    result = process_input("")
    assert result["blocked"] is True


def test_mrn_redacted():
    result = process_input("Patient MRN: 1234567 needs prior auth")
    assert result["phi_redacted"] is True
    assert "MRN" in result["phi_detected"]


def test_email_redacted():
    result = process_input(
        "Contact me at john.doe@hospital.com about my coverage"
    )
    assert result["phi_redacted"] is True
    assert "EMAIL" in result["phi_detected"]


def test_message_too_long_blocked():
    result = process_input("x" * 2001)
    assert result["blocked"] is True


def test_clean_clinical_question():
    result = process_input(
        "What are the prior auth criteria for knee replacement?"
    )
    assert result["passed_guardrails"] is True
    assert result["phi_redacted"] is False
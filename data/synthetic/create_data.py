"""
Creates synthetic healthcare policy documents.
These are FICTIONAL insurance policies — not real.
Used as the RAG knowledge base for the agent.
In production: replace with real policy PDFs from your insurer.
"""

# Synthetic policy document 1 — General coverage
POLICY_GENERAL = """
BLUECROSS SHIELD GOLD PPO — MEMBER BENEFITS GUIDE 2024
(SYNTHETIC DATA — FOR DEMONSTRATION PURPOSES ONLY)

SECTION 1: UNDERSTANDING YOUR BENEFITS

1.1 DEDUCTIBLE
Your annual deductible is $1,500 for individual coverage and $3,000
for family coverage. The deductible is the amount you pay for covered
health care services before your insurance plan starts to pay.
Example: If you have a $1,500 deductible, you pay the first $1,500
of covered services yourself.

1.2 COPAYMENT (COPAY)
A copay is a fixed amount you pay for a covered health care service,
usually at the time you receive the service.
- Primary Care Visit: $25 copay
- Specialist Visit: $50 copay
- Urgent Care: $75 copay
- Emergency Room: $250 copay (waived if admitted)

1.3 COINSURANCE
After you meet your deductible, you pay a percentage of costs.
This plan uses 80/20 coinsurance:
- Insurance pays: 80% of covered services
- You pay: 20% of covered services
- Example: $1,000 procedure → you pay $200 after deductible

1.4 OUT-OF-POCKET MAXIMUM
$4,000 individual / $8,000 family per calendar year.
Once you reach this limit, insurance pays 100% of covered services.
Includes: deductibles, copays, and coinsurance.

1.5 PREMIUM
Your monthly premium is $450 for individual coverage.
Premiums do NOT count toward your deductible or out-of-pocket maximum.

SECTION 2: COVERED SERVICES

2.1 PREVENTIVE CARE (covered at 100%, no cost to you)
- Annual wellness exam
- Recommended immunizations
- Cancer screenings (mammogram, colonoscopy, PSA)
- Blood pressure, cholesterol, diabetes screenings

2.2 PRIMARY CARE
Covered after $25 copay per visit.
Includes: sick visits, chronic disease management, referrals.

2.3 SPECIALIST CARE
Covered after $50 copay per visit.
Referral from primary care physician required for most specialists.

2.4 MENTAL HEALTH AND SUBSTANCE USE
Covered same as medical/surgical benefits (mental health parity).
- Outpatient therapy: $25 copay per session
- Inpatient psychiatric: covered after deductible + 20% coinsurance
- Up to 30 inpatient days per year

2.5 PRESCRIPTION DRUGS
Tier 1 (Generic): $10 copay
Tier 2 (Preferred Brand): $35 copay
Tier 3 (Non-Preferred Brand): $70 copay
Tier 4 (Specialty): 20% coinsurance, max $200 per prescription

SECTION 3: PRIOR AUTHORIZATION

3.1 WHAT REQUIRES PRIOR AUTHORIZATION
The following services require prior authorization (prior approval)
before they are provided:
- Inpatient hospital admissions (non-emergency)
- Outpatient surgery
- Advanced imaging: MRI, CT scan, PET scan
- Physical therapy (after initial evaluation)
- Durable medical equipment over $500
- Specialty medications (Tier 3 and Tier 4)
- Home health care
- Skilled nursing facility care

3.2 HOW TO REQUEST PRIOR AUTHORIZATION
1. Your provider submits a prior authorization request
2. Include: diagnosis code (ICD-10), procedure code (CPT),
   clinical notes, medical necessity documentation
3. Standard review: decision within 3 business days
4. Urgent review: decision within 24 hours
5. You and your provider will be notified of the decision

3.3 PRIOR AUTHORIZATION APPEAL PROCESS
If prior authorization is denied:
Level 1 Appeal: Submit within 180 days of denial
  - Include: additional clinical documentation
  - Decision within 30 days (standard) or 72 hours (urgent)
Level 2 Appeal: External review by independent organization
  - Request within 60 days of Level 1 denial
  - Decision within 45 days
"""

# Synthetic policy document 2 — Specific procedures
POLICY_PROCEDURES = """
BLUECROSS SHIELD GOLD PPO — MEDICAL NECESSITY CRITERIA 2024
(SYNTHETIC DATA — FOR DEMONSTRATION PURPOSES ONLY)

SECTION 4: PHYSICAL THERAPY

4.1 COVERAGE
Physical therapy is covered when medically necessary.
- Requires prior authorization after initial evaluation visit
- Maximum: 30 visits per calendar year
- Additional visits available with medical necessity documentation

4.2 MEDICAL NECESSITY CRITERIA FOR PHYSICAL THERAPY
Physical therapy is considered medically necessary when:
a) Prescribed by a licensed physician
b) Patient has a condition that is expected to improve with therapy
c) Goals are measurable and achievable within a reasonable time
d) Patient is able to actively participate in therapy
e) Services cannot be provided through home exercise program alone

4.3 DOCUMENTATION REQUIRED
- Physician referral or prescription
- Diagnosis (ICD-10 code)
- Treatment plan with specific goals
- Expected duration of treatment
- Functional limitations documented

SECTION 5: ORTHOPEDIC PROCEDURES

5.1 TOTAL KNEE ARTHROPLASTY (CPT 27447)

Medical Necessity Criteria — ALL of the following must be met:
a) Diagnosis of moderate to severe osteoarthritis confirmed by X-ray
   (Kellgren-Lawrence grade 3 or 4)
b) Significant functional impairment documented
   (KOOS score less than 45, or equivalent validated tool)
c) Failure of conservative treatment for minimum 3 months:
   - Physical therapy (minimum 6 weeks)
   - NSAIDs or other anti-inflammatory medications
   - Corticosteroid injections (if appropriate)
d) BMI less than 40 (or structured weight management plan documented)
e) No active joint infection
f) Medically stable for surgical procedure

Prior Authorization Required: YES
Required Documentation:
- X-ray reports showing grade 3-4 osteoarthritis
- KOOS score or functional assessment
- Physical therapy records (6+ weeks)
- Trial of NSAIDs documented
- Surgeon's operative plan

5.2 TOTAL HIP ARTHROPLASTY (CPT 27130)

Medical Necessity Criteria:
a) Diagnosis of hip osteoarthritis, avascular necrosis, or
   rheumatoid arthritis confirmed by imaging
b) Significant pain and functional limitation
c) Failure of conservative treatment minimum 3 months
d) BMI less than 40

Prior Authorization Required: YES

SECTION 6: MEDICATIONS — PRIOR AUTH REQUIREMENTS

6.1 SPECIALTY MEDICATIONS REQUIRING PRIOR AUTH

Biologics for Rheumatoid Arthritis (e.g., adalimumab, etanercept):
Step Therapy Required:
Step 1: Trial of conventional DMARDs (methotrexate, hydroxychloroquine)
        Minimum 3 months at therapeutic dose
Step 2: If Step 1 fails, biologic therapy approved
Required Documentation:
- Rheumatology diagnosis confirmed
- Previous DMARD trial documented with dates and doses
- Disease activity score (DAS28 or equivalent)

Diabetes Medications (GLP-1 agonists, e.g., semaglutide):
Prior auth required for:
- Weight management indication (non-diabetic use)
- When used with other diabetes medications
Criteria: HbA1c greater than 7.5%, trial of metformin documented

SECTION 7: EMERGENCY AND URGENT CARE

7.1 EMERGENCY CARE
Covered at any hospital, in-network or out-of-network.
- Emergency room copay: $250 (waived if admitted)
- No prior authorization required for true emergencies
- Definition: sudden onset of symptoms that could result in
  serious harm if not treated immediately

7.2 URGENT CARE
- In-network urgent care: $75 copay
- Out-of-network urgent care: covered at 60% after deductible

7.3 AMBULANCE
- Emergency ambulance: covered at 80% after deductible
- Non-emergency ambulance: requires prior authorization
"""

# Synthetic document 3 — Drug formulary
DRUG_FORMULARY = """
BLUECROSS SHIELD GOLD PPO — DRUG FORMULARY 2024
(SYNTHETIC DATA — FOR DEMONSTRATION PURPOSES ONLY)

SECTION 8: PRESCRIPTION DRUG COVERAGE

8.1 FORMULARY TIERS

TIER 1 — GENERIC DRUGS ($10 copay, 30-day supply)
Common examples:
- Metformin (diabetes)
- Lisinopril (blood pressure)
- Atorvastatin (cholesterol)
- Omeprazole (acid reflux)
- Amlodipine (blood pressure)
- Sertraline (depression/anxiety)
- Levothyroxine (thyroid)

TIER 2 — PREFERRED BRAND ($35 copay, 30-day supply)
Examples:
- Januvia (diabetes)
- Eliquis (blood thinner)
- Xarelto (blood thinner)
- Symbicort (asthma/COPD)

TIER 3 — NON-PREFERRED BRAND ($70 copay, 30-day supply)
Prior authorization may be required.

TIER 4 — SPECIALTY ($200 max copay or 20% coinsurance)
Includes biologics, oncology drugs, and specialty medications.
ALL Tier 4 medications require prior authorization.

8.2 DRUG INTERACTIONS — COMMON COMBINATIONS TO REVIEW

Metformin + Lisinopril:
- Generally safe combination
- Both commonly used in diabetic patients with hypertension
- Monitor: kidney function (eGFR), potassium levels
- No significant pharmacokinetic interaction
- Lisinopril may have renoprotective benefit in diabetic nephropathy

Metformin contraindications:
- eGFR less than 30: contraindicated
- eGFR 30-45: use with caution, reduce dose
- Hold before contrast procedures, restart 48 hours after

Warfarin interactions (significant — always verify):
- Increases INR: antibiotics, NSAIDs, aspirin, amiodarone
- Decreases INR: rifampin, carbamazepine, St. John's Wort
- Monitoring: INR check within 3-5 days of any new medication

8.3 SPECIALTY DRUG PROGRAM
Specialty medications must be filled through:
- Preferred Specialty Pharmacy (in-network)
- 30-day supply limit (no 90-day supply for specialty drugs)
- Requires enrollment in specialty pharmacy program
- Nurse case manager assigned for complex medications
"""

def save_documents():
    """Save all synthetic documents as text files."""
    import os
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)

    docs = {
        "policy_general.txt": POLICY_GENERAL,
        "policy_procedures.txt": POLICY_PROCEDURES,
        "drug_formulary.txt": DRUG_FORMULARY,
    }

    for filename, content in docs.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {filepath} ({len(content)} chars)")

    print(f"\nTotal documents: {len(docs)}")
    print("These are the knowledge base files the agent answers from.")


if __name__ == "__main__":
    save_documents()
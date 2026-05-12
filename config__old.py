import os
from dotenv import load_dotenv

load_dotenv()

# Vertex AI Settings
VERTEX_PROJECT_ID    = "project-8cf22686-23da-490c-926"
VERTEX_LOCATION      = "us-central1" # Ensure this matches your project's region
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "project-8cf22686-23da-490c-926-b8b25c6d2a77.json")

EMBEDDING_MODEL      = "text-embedding-004"
LLM_MODEL            = "gemini-2.5-flash"
CHROMA_PERSIST_DIR   = "./chroma_db"
CHROMA_COLLECTION    = "tenders"
BM25_INDEX_PATH      = "./bm25_index.pkl"

# Langfuse Configuration
# Note: Langfuse needs to be initialized via os.environ or Langfuse client directly
# Ensure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST are in your .env file
# e.g., os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."

# Chunking
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
SEPARATORS    = ["\n\n", "\n", ".", " "]

# Retrieval
TOP_K_DENSE   = 60
TOP_K_SPARSE  = 40
TOP_K_FINAL   = 100
TOP_K_RERANK  = 30

# Pre-defined queries for document checklist
DOCUMENT_QUERIES = [
    "einzureichende Unterlagen Angebot Nachweise",
    "Formblätter Anlagen beizufügen",
    "Eignungsnachweis vorzulegen einreichen",
    "Teilnahmebedingungen Unterlagen Pflicht",
    "Angebotsbestandteile Checkliste vollständig",
    "Formblatt E1 E2 E3 E4 E5 E6 Mantelbogen",
    "Pflichtangaben Bieter Eigenerklärung",
    "Handelsregister Betriebshaftpflicht Versicherung Nachweis",
]

# Field definitions matching the target JSON structure
FIELDS = [
    # ── Contracting Authority ──────────────────────────────────────
    {
        "id": "contracting_authority",
        "path": ["Contracting Authority"],
        "queries": [
            "Auftraggeber Name Vergabestelle Vergabenummer",
            "Öffentlicher Auftraggeber Bezeichnung",
        ],
        "instruction": "Full legal name of the contracting authority and the procurement/tender number (Vergabenummer) if available. Format: 'Name, Vergabenummer ...'",
        "extraction_mode": "narrative",
        "mandatory": True,
        "type": "string",
    },

    # ── Project Description ──────────────────────────────────────────
    {
        "id": "project_description",
        "path": ["Project Description"],
        "queries": [
            "Gegenstand der Ausschreibung Zusammenfassung",
            "Leistungsbeschreibung Überblick Vorhaben",
            "Zweck der Beschaffung Ziel des Projekts",
        ],
        "instruction": "Comprehensive summary of the tender's subject matter, goals, lots, and scope. Capture details about employees per year, hours, tasks, and locations. Provide a long, detailed paragraph. DO NOT abbreviate.",
        "extraction_mode": "narrative",
        "mandatory": True,
        "type": "string",
    },

    # ── Deadlines ──────────────────────────────────────────────
    {
        "id": "submission_deadline",
        "path": ["Submission Deadline"],
        "queries": [
            "Abgabefrist Angebot Datum Uhrzeit",
            "Einreichungsfrist Ende",
        ],
        "instruction": "The exact date and time for bid submission. Format: 'DD.MM.YYYY, HH:MM Uhr'",
        "extraction_mode": "exact",
        "mandatory": True,
        "type": "string",
    },
    {
        "id": "important_dates",
        "path": ["Important Dates"],
        "queries": [
            "Angebote und Bewertung Frist Bieterfragen Bindefrist Vertragsbeginn",
            "Termine Meilensteine Laufzeit Ende",
        ],
        "instruction": "Extract all relevant dates mentioned in sections like 'Angebote und Bewertung' or 'Fristen'. Include Question Deadline (Bieterfragen), Binding Deadline (Bindefrist), Start/End dates (Auftragsdauer), and any extensions. Format as a multi-line string: 'Label: Date'. Be thorough.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Scope & Requirements ──────────────────────────────────────
    {
        "id": "scope_and_requirements",
        "path": ["Scope & Requirements", "Scope & Requirements"],
        "queries": ["Scope and requirements general summary"],
        "instruction": "General summary of scope and requirements. If redundant, return null.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "contract_volume",
        "path": ["Scope & Requirements", "Contract Volume"],
        "queries": ["Auftragsvolumen Stunden Budget Rahmenvertrag"],
        "instruction": "Details about the contract volume, budget, or total hours.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "place_of_performance",
        "path": ["Scope & Requirements", "Place of Performance"],
        "queries": [
            "Ort der Leistungserbringung Projektstandort",
            "Remote Anteil München Hannover",
        ],
        "instruction": "Details about where the work is performed, remote policies, and specific office locations. Extract full sentences.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "standards_certifications",
        "path": ["Scope & Requirements", "Standards & Certifications"],
        "queries": [
            "Zertifizierungen Qualifikationen gefordert",
            "IPMA Prince2 ITIL Zertifikat",
        ],
        "instruction": "List all required or preferred certifications (ITIL, IPMA, Prince2, etc.).",
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "subcontracting_consortia",
        "path": ["Scope & Requirements", "Subcontracting & Consortia"],
        "queries": [
            "Subunternehmer Nachunternehmer Bietergemeinschaften Zulässigkeit",
            "Einsatz von Dritten Regelungen",
        ],
        "instruction": "Rules regarding subcontractors and consortia. Capture details about approvals, liability, and DSGVO requirements. Provide a detailed narrative.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "forms_esubmission",
        "path": ["Scope & Requirements", "Forms & e-Submission"],
        "queries": [
            "Abgabe Portale Formblätter e-Vergabe",
            "Einzureichende Dokumente Links",
        ],
        "instruction": "Information about submission portals (URLs) and specific forms/templates to be used. Include specific URLs if found.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Supplier Eligibility ──────────────────────────────────────
    {
        "id": "required_documents",
        "path": ["Supplier Eligibility", "Offer Submission Documents"],
        "queries": DOCUMENT_QUERIES,
        "instruction": "List of all documents, declarations, and evidence to be submitted. Be exhaustive.",
        "extraction_mode": "list",
        "mandatory": True,
        "type": "array",
    },
    {
        "id": "list_of_documents",
        "path": ["Supplier Eligibility", "List of Documents"],
        "queries": ["List of documents summary"],
        "instruction": "General list of documents. If redundant with Offer Submission Documents, return null.",
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "economic_standing",
        "path": ["Supplier Eligibility", "Economic & Financial Standing", "Minimum Turnover"],
        "queries": [
            "Mindestumsatz Vorjahre Gesamtumsatz",
            "Finanzielle Leistungsfähigkeit",
        ],
        "instruction": "Requirements regarding turnover or financial standing.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "turnover_comparable",
        "path": ["Supplier Eligibility", "Economic & Financial Standing", "Turnover in Comparable Services"],
        "queries": [
            "Umsatz vergleichbare Leistungen Bereich",
        ],
        "instruction": "Requirements for turnover specifically in comparable service areas for the last 3 years.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "legal_registration",
        "path": ["Supplier Eligibility", "Legal & Registration", "Trade/Professional Register Entry"],
        "queries": [
            "Handelsregisterauszug Berufsregister Eintragung",
        ],
        "instruction": "Requirements for registry entries (Handelsregister).",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "self_declarations",
        "path": ["Supplier Eligibility", "Legal & Registration", "Self-declarations (GWB §§123/124 or equivalent)"],
        "queries": [
            "Eigenerklärung Ausschlussgründe § 123 GWB",
            "Restriktive Maßnahmen Russland Verordnung",
        ],
        "instruction": "Required self-declarations regarding exclusion grounds or sanctions.",
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "subcontractor_identification",
        "path": ["Supplier Eligibility", "Legal & Registration", "Subcontractor Identification & Reliance"],
        "queries": ["Nachunternehmererklärung Eignungsleihe Formblatt"],
        "instruction": "Declarations needed for subcontractors or reliance on third-party capacities.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Technical & Professional Ability ───────────────────────────
    {
        "id": "personnel_profiles",
        "path": ["Technical & Professional Ability", "Personnel Profiles"],
        "queries": ["Anforderungsprofile Lebensläufe CV"],
        "instruction": "General requirements for personnel profiles and CVs.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "headcount_staffing",
        "path": ["Technical & Professional Ability", "Headcount / Staffing"],
        "queries": [
            "Anzahl Mitarbeiter Kapazität Los 1 2 3",
            "Mitarbeiter pro Jahr Stundenvolumen",
        ],
        "instruction": "Staffing requirements per lot (Los 1, 2, 3), including number of employees and estimated hours per year.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "reference_projects_tech",
        "path": ["Technical & Professional Ability", "Reference Projects"],
        "queries": [
            "Referenzprojekte Erfahrungen Nachweise",
            "Mindestanforderungen Rollen Profile",
        ],
        "instruction": "Requirements for reference projects and role-specific experience. Provide a detailed narrative.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Company Referrals ──────────────────────────────────────────
    {
        "id": "company_referrals",
        "path": ["Company Referrals"],
        "queries": ["Unternehmensreferenzen Firmenreferenzen"],
        "instruction": "Requirements for overall company reference projects (Unternehmensreferenzen).",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Award Criteria ───────────────────────────────────────────────
    {
        "id": "award_criteria_narrative",
        "path": ["Award Criteria"],
        "queries": [
            "Zuschlagskriterien Bewertungsmethodik Gewichtung",
            "Leistungspunkte Preispunkte Kennzahl",
        ],
        "instruction": "Explanation of how bids are evaluated (price vs. quality weighting) and any minimum point requirements.",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
]

MANDATORY_FIELDS = [
    "contracting_authority",
    "project_description",
    "submission_deadline",
    "required_documents",
]

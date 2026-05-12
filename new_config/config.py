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

# ─────────────────────────────────────────────────────────────────────
# Shared routing rules — prepend this to every field-extraction prompt.
# This is the disambiguation layer that prevents the model from putting
# the same fact under two fields. Each field's `instruction` below also
# contains its own local "do NOT include → goes to <field>" guardrails;
# this preamble is the global policy.
# ─────────────────────────────────────────────────────────────────────
ROUTING_RULES = """\
Decision routing — use this to choose the correct field:
- WHAT must be delivered / HOW the service is performed
    → Scope & Requirements
- PERSONNEL skills / certifications / seniority required to prove capability
    → Technical & Professional Ability → Personnel Profiles
- FTE counts / team size / availability windows
    → Technical & Professional Ability → Headcount / Staffing
- Reference projects of the PERSONNEL (key persons, CV-attached references)
    → Technical & Professional Ability → Reference Projects
- Reference projects of the BIDDER FIRM (company-level Unternehmensreferenzen)
    → Company Referrals
- HOW the winner is chosen / weights / scoring formulas
    → Award Criteria
- Declarations or documents the bidder must SUBMIT for eligibility
    → Supplier Eligibility

Cross-cutting rules:
- If a fact could fit two fields, place it under the MORE SPECIFIC one and
  return null in the more general one. Never repeat the same fact verbatim
  across two fields.
- Personnel-held certs (PMP, ITIL, IPMA, Prince2 held by named roles) belong
  in Personnel Profiles, NOT in Standards & Certifications.
- Company-level certs (ISO 9001 of the bidder firm) belong in Supplier
  Eligibility documents, NOT in Standards & Certifications.
- Standards & Certifications is reserved for standards the SERVICE/DELIVERABLE
  itself must comply with (e.g. BSI Grundschutz, DSGVO, ISO 27001 of the
  delivered system).
- Output the SOURCE language. Preserve numbers, percentages, dates, role
  names, and named entities verbatim.
- If the requested information is not in the retrieved context, return null.
  Do NOT invent or paraphrase content that is not present.
"""

# Pre-defined queries for the bidder document checklist.
# Mix exact German form-language (good for BM25) with descriptive phrases
# (good for dense embedding retrieval).
DOCUMENT_QUERIES = [
    "einzureichende Unterlagen Angebot Nachweise vollständig",
    "Eignungsnachweise Bietererklärungen vorzulegen",
    "Eigenerklärung zur Eignung Ausschlussgründe",
    "Formblatt Mantelbogen Anlage Vorlage einreichen",
    "Nachweis der Befugnis Gewerbeberechtigung Handelsregister",
    "Nachweis der Zuverlässigkeit Strafregister Sozialversicherung Finanzamt",
    "finanzielle wirtschaftliche Leistungsfähigkeit Mindestumsatz Versicherung",
    "technische berufliche Leistungsfähigkeit Referenzen Personal",
    "Pflichtangaben Bieter Bietergemeinschaft Subunternehmer Erklärung",
    "Verpflichtungserklärung VO 2022/576 Russland Sanktion",
    "Angebotsbestandteile Checkliste Teilnahmeantrag",
]

# ─────────────────────────────────────────────────────────────────────
# Field definitions
#
# Query design: each query line targets ONE retrieval angle. Two angles
# are reliably useful in tender PDFs:
#   (a) exact German form-language (Vergabestelle, Mindestumsatz,
#       Nachunternehmererklärung) — what the document literally calls it,
#       optimised for BM25.
#   (b) descriptive paraphrase — what the section is conceptually about,
#       optimised for the dense retriever.
# Most queries below blend both within a single string so each line earns
# its keep on both BM25 and embedding sides.
#
# Instruction design: each instruction states (1) what to capture,
# (2) the output shape, (3) explicit DO-NOT-INCLUDE redirects for the
# fields that share a fuzzy boundary with this one.
# ─────────────────────────────────────────────────────────────────────
FIELDS = [
    # ── Contracting Authority ────────────────────────────────────────
    {
        "id": "contracting_authority",
        "path": ["Contracting Authority"],
        "queries": [
            "Auftraggeber Name Anschrift Vergabestelle",
            "Vergabenummer Aktenzeichen Bezeichnung Vergabeverfahren",
            "öffentlicher Auftraggeber Bedarfsträger Behörde",
            "Ansprechpartner Vergabe Kontakt eMail",
        ],
        "instruction": (
            "Identify the contracting authority awarding the contract. "
            "Capture: full legal name, address (street, postal code, city, "
            "country if non-DACH), and the procurement reference "
            "(Vergabenummer / Aktenzeichen) if stated. "
            "Include procurement contact channels (e-Vergabe support hotline, "
            "procurement e-mail) ONLY if they appear in the authority/contact "
            "section of the document. "
            "Format: 'Name, Anschrift, Vergabenummer …'. "
            "Do NOT include project subject matter (→ Project Description), "
            "contract value (→ Scope & Requirements → Contract Volume), or "
            "evaluation contacts unrelated to procurement."
        ),
        "extraction_mode": "narrative",
        "mandatory": True,
        "type": "string",
    },

    # ── Project Description ──────────────────────────────────────────
    {
        "id": "project_description",
        "path": ["Project Description"],
        "queries": [
            "Gegenstand der Ausschreibung Beschaffungsgegenstand Überblick",
            "Vergabeverfahren Verfahrensart Rahmenvereinbarung Lose Aufteilung",
            "Zielsetzung Zweck Hintergrund des Vorhabens Projekt",
            "Zusammenfassung Übersicht Ausschreibung",
            "Teilangebote zulässig pro Los Zuschlag Anzahl Auftragnehmer",
        ],
        "instruction": (
            "High-level orientation paragraph that lets a reader understand "
            "WHAT is being procured at a glance. Capture: "
            "(a) the contract type / procurement vehicle (Rahmenvereinbarung, "
            "Einzelauftrag, Bietergemeinschaften zulässig …), "
            "(b) the procedure type and stages (zweistufiges "
            "Verhandlungsverfahren, offenes Verfahren, Teilnahmewettbewerb "
            "mit nachgelagerter Angebotsphase …), "
            "(c) the lot structure: list each Los with its title only; state "
            "whether Teilangebote per Los or sub-Los are allowed, "
            "(d) overall scope in one or two sentences, "
            "(e) Liefer- und Leistungszeitraum (term + extension options) if "
            "stated alongside the high-level scope. "
            "This field is the ORIENTATION. Detailed task lists belong in "
            "Scope & Requirements → Scope & Requirements. Numeric volumes "
            "(PT, hours, EUR) belong in Scope & Requirements → Contract "
            "Volume. FTE / team size belongs in Technical & Professional "
            "Ability → Headcount / Staffing. Award weights belong in Award "
            "Criteria. "
            "Output: prose, lightly structured with bullets only for the lot "
            "list. Preserve source-document phrasing."
        ),
        "extraction_mode": "narrative",
        "mandatory": True,
        "type": "string",
    },

    # ── Deadlines ────────────────────────────────────────────────────
    {
        "id": "submission_deadline",
        "path": ["Submission Deadline"],
        "queries": [
            "Abgabefrist Angebotsfrist Datum Uhrzeit",
            "Frist Einreichung elektronisches Angebot Ende",
            "Schlusstermin Angebotsabgabe Teilnahmeantrag",
        ],
        "instruction": (
            "The single primary submission deadline for the bid "
            "(Angebotsfrist) or, in two-stage procedures, for the participation "
            "request (Teilnahmefrist) — whichever is the next-due deadline "
            "from the bidder's perspective. "
            "Format: 'DD.MM.YYYY, HH:MM Uhr'. If only a date is given, use "
            "'DD.MM.YYYY'. "
            "Other dates (Bindefrist, Zuschlagsfrist, Bieterfragen, "
            "Vertragsbeginn, voraussichtliche Verhandlungsrunden) belong in "
            "Important Dates, NOT here."
        ),
        "extraction_mode": "exact",
        "mandatory": True,
        "type": "string",
    },
    {
        "id": "important_dates",
        "path": ["Important Dates"],
        "queries": [
            "Bieterfragen Frist Rückfragen Anbieterfragen Stichtag",
            "Bindefrist Zuschlagsfrist Angebotsbindung",
            "Vertragsbeginn Vertragslaufzeit Auftragsdauer Verlängerung Option",
            "Termine Meilensteine Verhandlungsrunde Aufklärung",
            "Zeitplan Vergabeverfahren voraussichtliche Termine",
        ],
        "instruction": (
            "All schedule milestones OTHER than the primary submission "
            "deadline. Capture (when explicitly stated): "
            "Bieterfragen-Frist, Bindefrist, Zuschlagsfrist, Vertragsbeginn / "
            "Auftragsdauer / Vertragsende, Verlängerungsoptionen, "
            "voraussichtliche Verhandlungsrunden, voraussichtlicher Zeitpunkt "
            "der Auswahl der Bieter, Frist für die letztgültige "
            "Angebotslegung. "
            "Output as a multi-line string, one milestone per line in the "
            "form 'Label: DD.MM.YYYY [HH:MM Uhr]'. Preserve the document's "
            "own labels. "
            "Do NOT duplicate the primary submission deadline already captured "
            "in Submission Deadline."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Scope & Requirements ─────────────────────────────────────────
    {
        "id": "scope_and_requirements",
        "path": ["Scope & Requirements", "Scope & Requirements"],
        "queries": [
            "Leistungsbeschreibung Aufgaben Tätigkeiten Aktivitäten",
            "Anforderungen funktional technisch nicht-funktional",
            "Leistungsumfang zu erbringende Leistungen Pflichten Auftragnehmer",
            "Aufgabenstellung Tätigkeitsfelder pro Los Leistungskategorie",
            "Beschreibung der Leistung Gegenstand Detail",
        ],
        "instruction": (
            "The detailed scope of work — the services to be delivered and "
            "the requirements the service must meet. Use this field for "
            "anything answering 'WHAT must be delivered / HOW is the service "
            "performed'. "
            "For each Los (or at the overall level if no lots): "
            "- services and tasks to be delivered (Tätigkeiten, Aufgaben, "
            "  Aktivitäten), "
            "- functional and technical requirements, "
            "- in-scope tooling and technical environment when explicitly "
            "  stated as part of the service (e.g. SAP PO/IS, UiPath, ServiceNow). "
            "Format: nested bullets per Los → Aufgabengebiet → einzelne "
            "Tätigkeiten. Preserve verbatim task wording. "
            "Do NOT include: numeric volumes / PT / EUR (→ Contract Volume), "
            "physical location / remote share (→ Place of Performance), "
            "personnel qualifications (→ Technical & Professional Ability → "
            "Personnel Profiles), bidder reference projects (→ Company "
            "Referrals), award scoring (→ Award Criteria). "
            "If the entire detailed scope is delegated to attached annexes "
            "and the retrieved context only references them, return null."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "contract_volume",
        "path": ["Scope & Requirements", "Contract Volume"],
        "queries": [
            "Auftragsvolumen Auftragswert geschätztes Gesamtvolumen",
            "Personentage PT Stundenkontingent Mannstunden",
            "Abrufvolumen Rahmenvertrag maximales Volumen Mengengerüst",
            "geschätztes Budget Gesamtwert in Euro netto",
            "Leistungsumfang in Personentagen pro Los Leistungskategorie",
        ],
        "instruction": (
            "Numeric SIZE of the contract: total budget, total Personentage "
            "(PT), total hours, EUR amount, ceiling values. Per-Los and "
            "per-Leistungskategorie breakdowns are wanted here when the "
            "source provides them as a volumes table "
            "(e.g. 'Los 1 — Fachliches Anforderungsmanagement: 5208 PT'). "
            "Distinguish from Headcount / Staffing — Headcount is "
            "FTE / availability windows / team size; Contract Volume is "
            "absolute totals. If both appear interleaved in the source, "
            "split: totals here, FTE/team-size in Headcount. "
            "Format: short prose plus a per-Los volumes table in bullets if "
            "applicable. Numbers verbatim."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "place_of_performance",
        "path": ["Scope & Requirements", "Place of Performance"],
        "queries": [
            "Ort der Leistungserbringung Erfüllungsort Standort",
            "Vor-Ort-Pflicht Remote-Anteil Hybrid Präsenz",
            "Liegenschaft Dienststelle Niederlassung Einsatzort",
            "Reisetätigkeit Reisekosten Anfahrt erwartet",
        ],
        "instruction": (
            "Where the work is performed: physical office locations, "
            "remote/on-site policy, hybrid expectations, travel obligations, "
            "specific buildings or sites. Preserve full sentences from the "
            "source so quantitative remote shares (e.g. '60% remote möglich, "
            "40% vor Ort in München') survive verbatim. "
            "If the retrieved context says only 'siehe Leistungsverzeichnis' "
            "and no concrete location is stated, return null."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "standards_certifications",
        "path": ["Scope & Requirements", "Standards & Certifications"],
        "queries": [
            "Normen Standards Vorgaben Compliance Gesetz",
            "BSI Grundschutz IT-Sicherheit ISO 27001 Anforderung",
            "DSGVO Datenschutz Verarbeitung Auftragsverarbeitung",
            "Barrierefreiheit BITV WCAG Standard",
            "Qualitätsstandards Vorschriften an die Leistung",
        ],
        "instruction": (
            "Standards, norms, regulations, or certifications that the "
            "DELIVERED SERVICE / SYSTEM itself must comply with — e.g. BSI "
            "IT-Grundschutz, ISO 27001 of the delivered solution, DSGVO, "
            "BITV/WCAG, sector-specific regulations. "
            "Do NOT include: personnel-held certifications such as PMP, "
            "ITIL, IPMA, Prince2, Scrum Master held by named roles "
            "(→ Personnel Profiles), nor company-level certifications such "
            "as ISO 9001 of the bidder firm (→ Supplier Eligibility "
            "documents). "
            "Output: an array of strings, each one standard/regulation. If "
            "no service-level standards are stated, return an empty array."
        ),
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "subcontracting_consortia",
        "path": ["Scope & Requirements", "Subcontracting & Consortia"],
        "queries": [
            "Nachunternehmer Subunternehmer Zulässigkeit Regelungen",
            "Bietergemeinschaft Arbeitsgemeinschaft Konsortium zulässig",
            "Eignungsleihe Drittunternehmen Verpflichtung Kapazitäten",
            "Haftung gesamtschuldnerisch bei Nachunternehmern Subunternehmern",
            "Genehmigung Auftraggeber Wechsel Nachunternehmer DSGVO",
        ],
        "instruction": (
            "RULES AND POLICY about using subcontractors and forming "
            "consortia: whether they are admitted, conditions and approvals "
            "required, liability arrangements (joint-and-several / "
            "gesamtschuldnerisch), DSGVO obligations, restrictions on "
            "switching subcontractors, Eignungsleihe rules. "
            "Do NOT include: the specific declaration FORMS / "
            "Verpflichtungserklärungen the bidder must hand in — those go "
            "to Supplier Eligibility → Legal & Registration → Subcontractor "
            "Identification & Reliance. "
            "Output: detailed narrative."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "forms_esubmission",
        "path": ["Scope & Requirements", "Forms & e-Submission"],
        "queries": [
            "e-Vergabe Plattform Vergabemarktplatz DTVP Abgabe Portal",
            "elektronische Angebotsabgabe Tool Software Hersteller",
            "Einreichung über Portal URL Link Webadresse",
            "Format Signatur Verschlüsselung Angebotsabgabe",
        ],
        "instruction": (
            "The HOW of submission: name and URL of the submission portal "
            "(DTVP, Vergabemarktplatz des Bundes, evergabe-online, "
            "ANKÖ, …), required signature/encryption format, links to "
            "the master document index if cited inline. "
            "Do NOT include the document checklist itself "
            "(→ Supplier Eligibility → Offer Submission Documents). "
            "Output: short narrative; quote URLs verbatim."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Supplier Eligibility ─────────────────────────────────────────
    {
        "id": "required_documents",
        "path": ["Supplier Eligibility", "Offer Submission Documents"],
        "queries": DOCUMENT_QUERIES,
        "instruction": (
            "Exhaustive list of every document, form, declaration, certificate, "
            "and proof the bidder must submit. Organise the list by the "
            "document's own categories where possible: "
            "Teilnahmeantrag / Angebotsformular, Eigenerklärungen, "
            "Bietergemeinschafts-/Subunternehmer-Erklärungen, Nachweis der "
            "Befugnis (Gewerbeberechtigung, Handelsregisterauszug), Nachweis "
            "der Zuverlässigkeit (Strafregister, Sozialversicherung, "
            "Finanzamt, Insolvenz, Verbandsverantwortlichkeitsregister), "
            "finanzielle und wirtschaftliche Leistungsfähigkeit, technische "
            "Leistungsfähigkeit, Los-spezifische Anlagen, Konzeptpapiere für "
            "Bewertung. "
            "Output: array of strings. Each item is the document's name; "
            "prefix with category where it disambiguates "
            "(e.g. 'Befugnis: Auszug aus dem Firmenbuch'). "
            "List the document NAME, not the wording of the declaration "
            "itself. Numeric thresholds attached to a document (e.g. "
            "'Mindestumsatz: 1 Mio EUR netto p.a.') stay with the document "
            "item."
        ),
        "extraction_mode": "list",
        "mandatory": True,
        "type": "array",
    },
    {
        "id": "list_of_documents",
        "path": ["Supplier Eligibility", "List of Documents"],
        "queries": [
            "Übersicht alle Anlagen Dokumente Teilnahmeantrag",
            "Index Dokumentenverzeichnis Vergabeunterlagen",
        ],
        "instruction": (
            "A general document list ONLY if the source provides a separate "
            "document index/inventory that is meaningfully different from "
            "the eligibility checklist already extracted into "
            "'Offer Submission Documents'. "
            "If the content would duplicate Offer Submission Documents, "
            "return null. Do not duplicate items between the two fields."
        ),
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "economic_standing",
        "path": ["Supplier Eligibility", "Economic & Financial Standing", "Minimum Turnover"],
        "queries": [
            "Mindestumsatz Gesamtumsatz drei Geschäftsjahre netto",
            "Jahresumsatz Mindestjahresumsatz Bieter",
            "wirtschaftliche finanzielle Leistungsfähigkeit Schwellenwert",
            "Betriebshaftpflichtversicherung Deckungssumme",
        ],
        "instruction": (
            "Minimum TOTAL turnover thresholds and other financial-standing "
            "requirements: minimum yearly turnover, look-back window "
            "(typically last 3 financial years), required liability "
            "insurance and its minimum coverage. Preserve thresholds "
            "verbatim ('zumindest 1 Mio EUR netto pro Jahr in den letzten "
            "drei Geschäftsjahren'). "
            "Turnover specifically in COMPARABLE/SIMILAR services goes to "
            "the next field, not here."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "turnover_comparable",
        "path": ["Supplier Eligibility", "Economic & Financial Standing", "Turnover in Comparable Services"],
        "queries": [
            "Umsatz vergleichbare Leistungen einschlägig Branche",
            "Mindestumsatz im Bereich Spezialgebiet Tätigkeitsfeld",
            "Branchenumsatz EAM-Beratung IT-Beratung drei Jahre",
        ],
        "instruction": (
            "Minimum turnover specifically in services COMPARABLE to the "
            "tender subject (e.g. 'mindestens 300.000 EUR netto im Bereich "
            "EAM-Beratung in den letzten drei abgeschlossenen "
            "Geschäftsjahren'). Always keep the thematic qualifier verbatim "
            "— it is the whole point of this field. "
            "If only a generic Gesamtumsatz is required, return null and "
            "leave that fact in Minimum Turnover."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "legal_registration",
        "path": ["Supplier Eligibility", "Legal & Registration", "Trade/Professional Register Entry"],
        "queries": [
            "Handelsregisterauszug Berufsregisterauszug Eintragung",
            "Gewerbeberechtigung Befugnis Nachweis",
            "Firmenbuchauszug Firmenregister Auszug aktuell",
        ],
        "instruction": (
            "Requirements for trade or professional register entries the "
            "bidder must evidence: Handelsregisterauszug, Firmenbuchauszug, "
            "Gewerbeberechtigung / Berufsregistereintrag, recency rules "
            "('nicht älter als 6 Monate'). "
            "Output: short narrative."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "self_declarations",
        "path": ["Supplier Eligibility", "Legal & Registration", "Self-declarations (GWB §§123/124 or equivalent)"],
        "queries": [
            "Eigenerklärung Ausschlussgründe § 123 § 124 GWB",
            "Eigenerklärung VO 2022/576 Russland Sanktion 833/2014",
            "Eigenerklärung Eignung Bieter Bewerber Pflicht",
            "Verbandsverantwortlichkeitsregister Strafregister Eigenerklärung",
            "restriktive Maßnahmen Sanktionsverordnung Eigenerklärung",
        ],
        "instruction": (
            "Required SELF-DECLARATIONS (Eigenerklärungen) regarding "
            "exclusion grounds, sanctions, or comparable legal compliance: "
            "GWB §§ 123/124 (DE), § 78 BVergG (AT), VO (EU) 2022/576 "
            "(Russia sanctions), VO 833/2014 Art. 5k. "
            "Output: array of strings, each item naming one declaration. "
            "Preserve the exact legal reference. Do NOT include the wording "
            "of the declaration itself — just its name and what it covers."
        ),
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },
    {
        "id": "subcontractor_identification",
        "path": ["Supplier Eligibility", "Legal & Registration", "Subcontractor Identification & Reliance"],
        "queries": [
            "Nachunternehmererklärung Subunternehmer Formblatt Pflicht",
            "Eignungsleihe Verpflichtungserklärung Drittunternehmen",
            "Liste Identifikation Nachunternehmer Bietergemeinschaft Erklärung",
            "Subunternehmerverzeichnis Drittfirma einzureichen",
        ],
        "instruction": (
            "DECLARATION FORMS the bidder must submit when relying on "
            "subcontractors or third-party capacities (Eignungsleihe). "
            "Examples: 'Verpflichtungserklärung von eingebundenen "
            "Drittunternehmen', 'Erklärung der Bietergemeinschaft', "
            "'Liste der Nachunternehmer'. "
            "This field is about WHICH FORMS must be filled out and handed "
            "in. The general POLICY on whether subcontractors are admitted, "
            "their liability, etc., goes to Scope & Requirements → "
            "Subcontracting & Consortia."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Technical & Professional Ability ─────────────────────────────
    {
        "id": "personnel_profiles",
        "path": ["Technical & Professional Ability", "Personnel Profiles"],
        "queries": [
            "Anforderungsprofil Rolle Schlüsselperson Mitarbeiter",
            "Qualifikation Berufserfahrung Senior Berater Architekt",
            "Personalprofil Projektmanager Entwickler Rolle pro Los",
            "Lebenslauf CV einzureichen Schlüsselpersonen Profile",
            "geforderte Zertifizierungen Personal PMP ITIL IPMA Prince2 Scrum",
            "Sprachkenntnisse Deutsch Englisch verhandlungssicher Rolle",
            "Mindestanforderung Bewertungskriterium Rolle [A] [B]",
        ],
        "instruction": (
            "QUALITATIVE requirements for the people the bidder will deploy. "
            "Per role and per Los where applicable, capture: "
            "- minimum years of experience (preserve thresholds verbatim, "
            "  e.g. 'mind. 5 Jahre'), "
            "- required CERTIFICATIONS HELD BY THE ROLE — PMP, IPMA Level D, "
            "  Prince2 Foundation, ITIL, Scrum Master, AWS Solutions "
            "  Architect (these are personnel-held certs and belong HERE, "
            "  NOT in Standards & Certifications), "
            "- required technical skills, languages, products, methodologies, "
            "- required language proficiency, "
            "- the document's own [A] Mindestanforderung vs "
            "  [B] Bewertungskriterium tags inline if used, "
            "- inline scoring tables placed at role level, verbatim "
            "  (e.g. '5–7 Jahre = 4 Punkte; 7–10 Jahre = 6 Punkte; "
            "  >10 Jahre = 10 Punkte'). "
            "Format: nested bullets, Los → Rolle → Anforderungen. "
            "Do NOT include: FTE counts / team size / availability windows "
            "(→ Headcount / Staffing); reference projects required of those "
            "personnel (→ Reference Projects); company-level references "
            "(→ Company Referrals); top-level Preis-vs-Qualität award split "
            "(→ Award Criteria — only the role-level scoring stays here)."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "headcount_staffing",
        "path": ["Technical & Professional Ability", "Headcount / Staffing"],
        "queries": [
            "Anzahl Mitarbeiter FTE Vollzeitäquivalent benötigt",
            "Teamgröße Mindestteam Kapazität pro Los",
            "Verfügbarkeit Bereitstellung Mitarbeiter Stundenkontingent",
            "Mindestanzahl Personal pro Rolle gleichzeitig verfügbar",
            "Personentage pro Mitarbeiter Jahr Auslastung",
        ],
        "instruction": (
            "QUANTITATIVE staffing requirements: required FTE counts, "
            "minimum team size, number of personnel per role per Los, "
            "availability windows ('während der gesamten Vertragslaufzeit "
            "verfügbar', 'innerhalb von 5 Werktagen abrufbar'), "
            "Stundenkontingent oder PT pro Mitarbeiter where stated as a "
            "staffing constraint (not a contract total). "
            "Distinguish from Contract Volume (absolute totals across the "
            "contract) — Headcount is the TEAM SHAPE the bidder must field. "
            "Do NOT include: per-role qualifications / certifications "
            "(→ Personnel Profiles)."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "reference_projects_tech",
        "path": ["Technical & Professional Ability", "Reference Projects"],
        "queries": [
            "Projektreferenzen Schlüsselpersonen einzelne Mitarbeiter",
            "persönliche Referenzen Lebenslauf belegen Rolle",
            "Referenzprojekte des benannten Personals nachweisen",
            "Einzelreferenz pro Rolle Mindestanzahl Zeitraum",
            "Erfahrung Schlüsselperson Beleg Projekt Auftraggeber",
        ],
        "instruction": (
            "PERSONNEL-LEVEL reference projects — reference projects required "
            "from the named key persons / role-holders themselves (typically "
            "attached to or referenced in their CVs). "
            "Capture: minimum NUMBER of references per role/person, "
            "look-back window (verbatim, e.g. 'letzte 36 Monate, "
            "zurückgerechnet vom Ende der Teilnahmefrist'), thematic "
            "similarity criteria, minimum project size if given (PT, EUR, "
            "duration), the person's required role on the reference project, "
            "named example projects if the source lists any. "
            "Do NOT include: company-level references of the bidder firm "
            "as such (→ Company Referrals). "
            "If the source bundles personnel and company references "
            "together with no separate personnel-level requirement, return "
            "null and let Company Referrals carry the content."
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Company Referrals ────────────────────────────────────────────
    {
        "id": "company_referrals",
        "path": ["Company Referrals"],
        "queries": [
            "Unternehmensreferenzen Firmenreferenzen Bieter",
            "Referenzprojekte des Unternehmens Auftragnehmer",
            "Bieterreferenzen Eignungsreferenzen vergleichbare Aufträge",
            "Mindestanzahl Referenzen Bieter letzte drei Jahre",
            "Auswahlreferenz Stufe 1 Punkte Bewertung",
        ],
        "instruction": (
            "COMPANY-LEVEL / BIDDER-FIRM reference projects — what the firm "
            "as an organisation has delivered. "
            "Capture: minimum NUMBER of references per Los, look-back "
            "window verbatim, thematic / domain coverage required "
            "(e.g. 'mindestens zwei verschiedene Bereiche aus IT-Architektur, "
            "Solution Design, IT-Security, New Technologies, Analytics & BI'), "
            "minimum project size (Auftragsvolumen in EUR, Personentage, "
            "Laufzeit), the firm's required role on the reference "
            "(Hauptauftragnehmer, Subunternehmer, Konsortialführer), "
            "right of the contracting authority to verify with prior "
            "clients, restrictions (projects older than X years not "
            "accepted), named example projects if listed, and "
            "Auswahlreferenz scoring if it sits at company level. "
            "Do NOT include: references attached to specific named persons "
            "(→ Technical & Professional Ability → Reference Projects). "
            "If all reference requirements in the source are bundled with "
            "the personnel requirements, output: 'Siehe Personnel Profiles.'"
        ),
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Award Criteria ───────────────────────────────────────────────
    {
        "id": "award_criteria_narrative",
        "path": ["Award Criteria"],
        "queries": [
            "Zuschlagskriterien Bewertungsmatrix Gewichtung Preis Qualität",
            "Wertungsmethode UfAB Richtwertmethode Leistung Preis Verhältnis",
            "Bestbieterprinzip wirtschaftlich technisch günstigstes Angebot",
            "Auswahlkriterien Stufe 1 Teilnahmewettbewerb Punkte",
            "Mindestpunktzahl K.O.-Kriterium Schwellenwert Wertung",
            "Preiskennzahl Leistungspunkte Kriterienkennzahl Formel",
        ],
        "instruction": (
            "How the winning bid is selected. Capture: "
            "(a) the awarding principle even when no breakdown is given "
            "(e.g. 'wirtschaftlich und technisch bestes Angebot', "
            "'Bestbieterprinzip', 'wirtschaftlichstes Angebot gem. § 58 VgV "
            "i.V.m. § 127 GWB'), "
            "(b) the top-level weighting (Preis vs Qualität in %), "
            "(c) the scoring formula verbatim if given (UfAB-Richtwertmethode, "
            "Z = L / P, Wertungskennzahl = Preiskennzahl + Kriterienkennzahl), "
            "(d) every sub-criterion with its weight or maximum points, "
            "preserving the source's hierarchy "
            "(e.g. '1.1 Konzept zum Projektmanagement 50 % = max. 50 Punkte'), "
            "(e) the scoring rubric verbatim, "
            "(f) for two-stage procedures: BOTH the Auswahlkriterien (Stufe 1) "
            "and the Zuschlagskriterien (Stufe 2), each clearly labelled "
            "with its stage, "
            "(g) any minimum point thresholds / K.O.-Kriterien. "
            "Do NOT include: per-role scoring tables that sit at role level "
            "in the source (→ Personnel Profiles). Only the global award "
            "rubric and any criteria-level point breakdowns belong here."
        ),
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

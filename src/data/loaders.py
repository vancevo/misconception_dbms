"""Per-source raw loaders for the ASAG Research Framework.

Each loader reads a specific dataset format and returns a list of
UnifiedRecord objects. Malformed rows are logged and skipped.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from src.data.schema import UnifiedRecord

logger = logging.getLogger(__name__)


# ── SciEntsBank Loader ────────────────────────────────────────────────


def load_scientsbank(data_dir: str | Path) -> list[UnifiedRecord]:
    """Load SciEntsBank dataset from XML/text files.

    Expected directory structure:
        data_dir/
            <split>/          (e.g., train, test_ua, test_uq, test_ud)
                *.xml         (one XML file per question or beetle-style XML)

    Each XML file is expected to contain question-answer pairs with 5-way
    labels. The exact XML schema follows the SemEval-2013 Task 7 format.

    Args:
        data_dir: Path to the SciEntsBank raw data directory.

    Returns:
        List of UnifiedRecord objects.
    """
    data_dir = Path(data_dir)
    records: list[UnifiedRecord] = []

    if not data_dir.exists():
        logger.warning("SciEntsBank directory not found: %s", data_dir)
        return records

    counter = 0

    # Walk through split directories
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        for xml_file in sorted(split_dir.glob("*.xml")):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning(
                    "Malformed XML in SciEntsBank file %s: %s — skipping",
                    xml_file, e,
                )
                continue

            # Parse SemEval-2013 Beetle/SciEntsBank XML format
            for question_elem in root.iter("question"):
                q_id = question_elem.get("id", "")
                q_text = question_elem.findtext("questionText", "")
                ref_answers = question_elem.findall(".//referenceAnswer")
                ref_answer_text = ""
                alt_refs: list[str] = []
                for i, ra in enumerate(ref_answers):
                    text = (ra.text or "").strip()
                    if i == 0:
                        ref_answer_text = text
                    else:
                        alt_refs.append(text)

                for sa_elem in question_elem.iter("studentAnswer"):
                    try:
                        counter += 1
                        sa_id = sa_elem.get("id", f"unknown_{counter}")
                        sa_text = (sa_elem.text or "").strip()
                        accuracy = sa_elem.get("accuracy", "unknown")

                        rec = UnifiedRecord(
                            sample_id=f"SEB_{counter:05d}",
                            source_dataset="scientsbank",
                            original_id=sa_id,
                            question_id=q_id,
                            domain="science",
                            subdomain=question_elem.get("module", "general"),
                            difficulty="unknown",
                            question=q_text,
                            reference_answer=ref_answer_text,
                            student_answer=sa_text,
                            alternative_reference_answers=alt_refs,
                            label_5way=accuracy,
                            split=split_name,
                            is_human_annotated=True,
                        )
                        records.append(rec)
                    except Exception as e:
                        logger.warning(
                            "Malformed SciEntsBank student answer in %s (id=%s): %s — skipping",
                            xml_file, sa_elem.get("id", "?"), e,
                        )

    logger.info("Loaded %d SciEntsBank records from %s", len(records), data_dir)
    return records



# ── MohlerASAG Loader ────────────────────────────────────────────────


def load_mohler(data_dir: str | Path) -> list[UnifiedRecord]:
    """Load MohlerASAG dataset from CSV/text files.

    The Mohler dataset typically has columns for question, answer, and
    multiple annotator scores. The loader averages annotator scores for
    score_raw and populates alternative_reference_answers when multiple
    reference answers exist per question.

    Args:
        data_dir: Path to the MohlerASAG raw data directory.

    Returns:
        List of UnifiedRecord objects.
    """
    data_dir = Path(data_dir)
    records: list[UnifiedRecord] = []

    if not data_dir.exists():
        logger.warning("MohlerASAG directory not found: %s", data_dir)
        return records

    # Look for CSV files in the directory
    csv_files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.txt"))
    if not csv_files:
        logger.warning("No CSV/text files found in MohlerASAG directory: %s", data_dir)
        return records

    counter = 0
    # Track reference answers per question for alternative_reference_answers
    question_ref_answers: dict[str, list[str]] = {}

    # First pass: collect all reference answers per question
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q_id = row.get("question_id", row.get("qid", ""))
                    ref = row.get("reference_answer", row.get("desired_answer", "")).strip()
                    if q_id and ref:
                        if q_id not in question_ref_answers:
                            question_ref_answers[q_id] = []
                        if ref not in question_ref_answers[q_id]:
                            question_ref_answers[q_id].append(ref)
        except Exception as e:
            logger.warning(
                "Error reading MohlerASAG file %s for ref answers: %s — skipping",
                csv_file, e,
            )

    # Second pass: build records
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):
                    try:
                        counter += 1
                        q_id = row.get("question_id", row.get("qid", f"q_{counter}"))
                        question = row.get("question", row.get("question_text", "")).strip()
                        ref_answer = row.get("reference_answer", row.get("desired_answer", "")).strip()
                        student_answer = row.get("student_answer", row.get("answer", "")).strip()

                        # Average annotator scores
                        score_cols = [
                            k for k in row.keys()
                            if k.startswith("score") or k.startswith("grade")
                            or k.startswith("me") or k.startswith("other")
                        ]
                        scores = []
                        for col in score_cols:
                            try:
                                scores.append(float(row[col]))
                            except (ValueError, TypeError):
                                pass

                        # Fall back to a single score column
                        if not scores:
                            for col in ["score_raw", "score", "avg_score"]:
                                if col in row:
                                    try:
                                        scores.append(float(row[col]))
                                    except (ValueError, TypeError):
                                        pass

                        score_raw = sum(scores) / len(scores) if scores else None

                        # Build alternative reference answers
                        alt_refs = [
                            r for r in question_ref_answers.get(q_id, [])
                            if r != ref_answer
                        ]

                        original_id = row.get("id", row.get("instance_id", f"mohler_{counter}"))

                        rec = UnifiedRecord(
                            sample_id=f"MOH_{counter:05d}",
                            source_dataset="mohler",
                            original_id=str(original_id),
                            question_id=str(q_id),
                            domain="computer_science",
                            subdomain=row.get("subdomain", row.get("topic", "general")),
                            difficulty="unknown",
                            question=question,
                            reference_answer=ref_answer,
                            student_answer=student_answer,
                            alternative_reference_answers=alt_refs,
                            score_raw=score_raw,
                            is_human_annotated=True,
                        )
                        records.append(rec)
                    except Exception as e:
                        logger.warning(
                            "Malformed MohlerASAG row %d in %s: %s — skipping",
                            row_num, csv_file, e,
                        )
        except Exception as e:
            logger.warning(
                "Error reading MohlerASAG file %s: %s — skipping",
                csv_file, e,
            )

    logger.info("Loaded %d MohlerASAG records from %s", len(records), data_dir)
    return records



# ── Data_Generate Loader ──────────────────────────────────────────────


def _safe_parse_list(value: str) -> list[str]:
    """Parse a string representation of a list, returning [] on failure."""
    if not value or value.strip() in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # Try JSON parsing as fallback
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            return []


def _safe_parse_dict_list(value: str) -> list[dict]:
    """Parse a string representation of a list of dicts, returning [] on failure."""
    if not value or value.strip() in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
        return []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [d for d in parsed if isinstance(d, dict)]
            return []
        except (json.JSONDecodeError, TypeError):
            return []


def _safe_float(value: str | None) -> float | None:
    """Convert a string to float, returning None on failure."""
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: str | None) -> str | None:
    """Convert a value to str or None if empty/nan."""
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    return str(value).strip()


def load_data_generate(csv_path: str | Path) -> list[UnifiedRecord]:
    """Load Data_Generate dataset from CSV.

    Parses all 30 columns and maps them to UnifiedRecord fields.
    Preserves adversarial_variant_of linkage and misconception_tags.

    Column mapping:
        instance_id → original_id
        question_id → question_id
        domain → domain
        subdomain → subdomain
        difficulty → difficulty
        split → split
        question → question
        reference_answer → reference_answer
        alternative_reference_answers → alternative_reference_answers (parsed list)
        key_concepts → key_concepts (parsed list)
        misconception_inventory → misconception_inventory (parsed list of dicts)
        student_answer → student_answer
        student_answer_style → student_answer_style
        lexical_overlap_level → (not mapped to schema, informational)
        semantic_correctness_score_0_5 → score_raw
        label_5way → label_5way
        label_3way → label_3way
        label_2way → label_2way
        misconception_tags → misconception_tags (parsed list)
        misconception_span_rationale → (stored in feedback context)
        missing_concepts → missing_concepts (parsed list)
        extra_incorrect_claims → extra_incorrect_claims (parsed list)
        feedback_short → feedback_short
        feedback_detailed → feedback_detailed
        feedback_type → feedback_type
        feedback_tone → feedback_tone
        adversarial_variant_of → adversarial_variant_of
        perturbation_type → perturbation_type
        robustness_notes → (not mapped, informational)
        annotation_confidence → annotation_confidence

    Args:
        csv_path: Path to the data-generate.csv file.

    Returns:
        List of UnifiedRecord objects.
    """
    csv_path = Path(csv_path)
    records: list[UnifiedRecord] = []

    if not csv_path.exists():
        logger.warning("Data_Generate CSV not found: %s", csv_path)
        return records

    # Increase CSV field size limit for large fields
    csv.field_size_limit(10 * 1024 * 1024)

    counter = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            try:
                counter += 1
                instance_id = row.get("instance_id", "").strip()
                score_raw = _safe_float(row.get("semantic_correctness_score_0_5"))
                annotation_conf = _safe_float(row.get("annotation_confidence"))

                # Determine if adversarial
                perturbation = _safe_str(row.get("perturbation_type"))
                adv_variant_of = _safe_str(row.get("adversarial_variant_of"))
                is_adversarial = perturbation is not None and perturbation != ""

                rec = UnifiedRecord(
                    sample_id=f"GEN_{counter:05d}",
                    source_dataset="data_generate",
                    original_id=instance_id,
                    question_id=row.get("question_id", "").strip(),
                    domain=row.get("domain", "").strip(),
                    subdomain=row.get("subdomain", "").strip(),
                    difficulty=row.get("difficulty", "unknown").strip(),
                    question=row.get("question", "").strip(),
                    reference_answer=row.get("reference_answer", "").strip(),
                    student_answer=row.get("student_answer", "").strip(),
                    alternative_reference_answers=_safe_parse_list(
                        row.get("alternative_reference_answers", "")
                    ),
                    key_concepts=_safe_parse_list(row.get("key_concepts", "")),
                    misconception_inventory=_safe_parse_dict_list(
                        row.get("misconception_inventory", "")
                    ),
                    misconception_tags=_safe_parse_list(
                        row.get("misconception_tags", "")
                    ),
                    missing_concepts=_safe_parse_list(
                        row.get("missing_concepts", "")
                    ),
                    extra_incorrect_claims=_safe_parse_list(
                        row.get("extra_incorrect_claims", "")
                    ),
                    score_raw=score_raw,
                    label_5way=_safe_str(row.get("label_5way")),
                    label_3way=_safe_str(row.get("label_3way")),
                    label_2way=_safe_str(row.get("label_2way")),
                    feedback_short=_safe_str(row.get("feedback_short")),
                    feedback_detailed=_safe_str(row.get("feedback_detailed")),
                    feedback_type=_safe_str(row.get("feedback_type")),
                    feedback_tone=_safe_str(row.get("feedback_tone")),
                    split=row.get("split", "").strip(),
                    is_synthetic=True,
                    is_adversarial=is_adversarial,
                    perturbation_type=perturbation,
                    adversarial_variant_of=adv_variant_of,
                    student_answer_style=_safe_str(row.get("student_answer_style")),
                    annotation_confidence=annotation_conf,
                    is_human_annotated=False,
                    usable_for_grading=True,
                    usable_for_feedback=True,
                    usable_for_misconception_mining=True,
                    usable_for_robustness_eval=True,
                )
                records.append(rec)
            except Exception as e:
                logger.warning(
                    "Malformed Data_Generate row %d (id=%s): %s — skipping",
                    row_num, row.get("instance_id", "?"), e,
                )

    logger.info("Loaded %d Data_Generate records from %s", len(records), csv_path)
    return records



# ── Data_Scraping Loader ──────────────────────────────────────────────


def load_data_scraping(json_path: str | Path) -> list[UnifiedRecord]:
    """Load Data_Scraping dataset from JSON.

    All records have empty student_answer fields and all four usability
    flags set to false. is_synthetic is set to false.

    JSON structure (array of objects):
        {
            "id": "college-physics-2e_1_1",
            "questions": "...",
            "reference_answer": "...",
            "student_answer": "",
            "label": "openstax_college-physics-2e"
        }

    The "label" field is used as domain (the OpenStax textbook identifier).

    Args:
        json_path: Path to the data-scraping.json file.

    Returns:
        List of UnifiedRecord objects.
    """
    json_path = Path(json_path)
    records: list[UnifiedRecord] = []

    if not json_path.exists():
        logger.warning("Data_Scraping JSON not found: %s", json_path)
        return records

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error reading Data_Scraping JSON %s: %s", json_path, e)
        return records

    if not isinstance(data, list):
        logger.warning("Data_Scraping JSON is not a list: %s", json_path)
        return records

    for idx, entry in enumerate(data):
        try:
            original_id = str(entry.get("id", f"scr_{idx}"))
            # Derive question_id from the original_id (e.g., "college-physics-2e_1_1")
            question_id = original_id

            # The "label" field contains the textbook/domain identifier
            domain_label = str(entry.get("label", "unknown"))
            # Extract subdomain from the domain label (e.g., "openstax_college-physics-2e" → "college-physics-2e")
            subdomain = domain_label.replace("openstax_", "") if domain_label.startswith("openstax_") else domain_label

            rec = UnifiedRecord(
                sample_id=f"SCR_{idx + 1:05d}",
                source_dataset="data_scraping",
                original_id=original_id,
                question_id=question_id,
                domain=domain_label,
                subdomain=subdomain,
                difficulty="unknown",
                question=str(entry.get("questions", "")).strip(),
                reference_answer=str(entry.get("reference_answer", "")).strip(),
                student_answer=str(entry.get("student_answer", "")).strip(),
                is_synthetic=False,
                is_human_annotated=False,
                usable_for_grading=False,
                usable_for_feedback=False,
                usable_for_misconception_mining=False,
                usable_for_robustness_eval=False,
            )
            records.append(rec)
        except Exception as e:
            logger.warning(
                "Malformed Data_Scraping entry %d (id=%s): %s — skipping",
                idx, entry.get("id", "?"), e,
            )

    logger.info("Loaded %d Data_Scraping records from %s", len(records), json_path)
    return records

from __future__ import annotations

import re
import io
import math
from pathlib import Path
from typing import Optional

import easyocr
import numpy as np
from PIL import Image

from models.schemas import StructuredLabResult, LabParameter
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


PARAM_ALIASES: dict[str, str] = {
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hct": "hematocrit",
    "haematocrit": "hematocrit",
    "hematocrit": "hematocrit",
    "rbc": "rbc_count",
    "red blood cell": "rbc_count",
    "red blood cells": "rbc_count",
    "wbc": "wbc_count",
    "white blood cell": "wbc_count",
    "white blood cells": "wbc_count",
    "leukocyte": "wbc_count",
    "plt": "platelet_count",
    "platelet": "platelet_count",
    "platelets": "platelet_count",
    "thrombocyte": "platelet_count",
    "mcv": "mcv",
    "mch": "mch",
    "mchc": "mchc",
    "glu": "glucose",
    "glucose": "glucose",
    "blood glucose": "glucose",
    "fbs": "fasting_blood_sugar",
    "fasting blood sugar": "fasting_blood_sugar",
    "hba1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "creat": "creatinine",
    "creatinine": "creatinine",
    "bun": "blood_urea_nitrogen",
    "urea": "blood_urea_nitrogen",
    "ua": "uric_acid",
    "uric acid": "uric_acid",
    "chol": "cholesterol_total",
    "total cholesterol": "cholesterol_total",
    "ldl": "cholesterol_ldl",
    "hdl": "cholesterol_hdl",
    "tg": "triglycerides",
    "triglyceride": "triglycerides",
    "triglycerides": "triglycerides",
    "alt": "alt",
    "sgpt": "alt",
    "ast": "ast",
    "sgot": "ast",
    "alp": "alkaline_phosphatase",
    "ggt": "ggt",
    "tbil": "bilirubin_total",
    "total bilirubin": "bilirubin_total",
    "dbil": "bilirubin_direct",
    "tsh": "tsh",
    "ft4": "free_t4",
    "ft3": "free_t3",
    "na": "sodium",
    "sodium": "sodium",
    "k": "potassium",
    "potassium": "potassium",
    "cl": "chloride",
    "chloride": "chloride",
    "ca": "calcium",
    "calcium": "calcium",
    "vit d": "vitamin_d",
    "vitamin d": "vitamin_d",
    "25-oh-d": "vitamin_d",
    "vit b12": "vitamin_b12",
    "vitamin b12": "vitamin_b12",
    "crp": "crp",
    "c-reactive protein": "crp",
    "esr": "esr",
}

NORMAL_RANGES: dict[str, tuple[float, float, str]] = {
    "hemoglobin":          (12.0, 17.5, "g/dl"),
    "hematocrit":          (36.0, 52.0, "%"),
    "rbc_count":           (3.8,   5.8, "10^6/ul"),
    "wbc_count":           (4.0,  11.0, "10^3/ul"),
    "platelet_count":      (150,   400, "10^3/ul"),
    "glucose":             (70,    100, "mg/dl"),
    "fasting_blood_sugar": (70,    100, "mg/dl"),
    "hba1c":               (4.0,   5.7, "%"),
    "creatinine":          (0.6,   1.2, "mg/dl"),
    "cholesterol_total":   (0,     200, "mg/dl"),
    "cholesterol_ldl":     (0,     100, "mg/dl"),
    "cholesterol_hdl":     (40,   9999, "mg/dl"),
    "triglycerides":       (0,     150, "mg/dl"),
    "tsh":                 (0.4,   4.0, "miu/l"),
    "sodium":              (136,   145, "meq/l"),
    "potassium":           (3.5,   5.0, "meq/l"),
}

UNIT_MAP: dict[str, str] = {
    "g/dl": "g/dl",
    "g/dL": "g/dl",
    "gm/dl": "g/dl",
    "mg/dl": "mg/dl",
    "mg/dL": "mg/dl",
    "mmol/l": "mmol/l",
    "mmol/L": "mmol/l",
    "iu/l": "iu/l",
    "IU/L": "iu/l",
    "u/l": "u/l",
    "U/L": "u/l",
    "miu/l": "miu/l",
    "mIU/L": "miu/l",
    "meq/l": "meq/l",
    "mEq/L": "meq/l",
    "%": "%",
    "10^3/ul": "10^3/ul",
    "10^3/µl": "10^3/ul",
    "10^6/ul": "10^6/ul",
    "10^6/µl": "10^6/ul",
    "cells/ul": "cells/ul",
    "pg": "pg",
    "fl": "fl",
    "ng/ml": "ng/ml",
    "ng/mL": "ng/ml",
    "pmol/l": "pmol/l",
}

VALUE_UNIT_RE = re.compile(
    r"(?P<modifier>[<>≤≥]?)\s*"
    r"(?P<value>\d{1,6}(?:[.,]\d{1,4})?)"
    r"(?:\s*(?P<unit>g/d[Ll]|mg/d[Ll]|mmol/[Ll]|mIU/[Ll]|miu/l|IU/[Ll]|u/[Ll]|U/[Ll]|"
    r"meq/[Ll]|mEq/[Ll]|10\^[36]/[uµ]l|cells/[uµ]l|ng/m[Ll]|pmol/[Ll]|%|fl|pg))?",
    re.IGNORECASE,
)


class OCRService:

    def __init__(self) -> None:
        logger.info("Initialising EasyOCR reader …")
        self._reader = easyocr.Reader(
            settings.OCR_LANGUAGES,
            gpu=settings.OCR_GPU,
            verbose=False,
        )
        logger.info("EasyOCR ready.")

    async def process_image(
        self,
        image_bytes: bytes,
        filename: str = "",
    ) -> tuple[str, StructuredLabResult | None, float]:
        raw_text, confidence = self._run_ocr(image_bytes)
        clean_text = self._clean_text(raw_text)
        structured = self._extract_lab_parameters(clean_text)
        return clean_text, structured, confidence

    async def process_pdf_pages(
        self,
        pages_images: list[bytes],
    ) -> tuple[str, StructuredLabResult | None, float]:
        all_text_parts = []
        confidences = []
        for page_bytes in pages_images:
            raw, conf = self._run_ocr(page_bytes)
            all_text_parts.append(raw)
            confidences.append(conf)

        merged_raw = "\n\n".join(all_text_parts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        clean_text = self._clean_text(merged_raw)
        structured = self._extract_lab_parameters(clean_text)
        return clean_text, structured, avg_conf

    def _run_ocr(self, image_bytes: bytes) -> tuple[str, float]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        results = self._reader.readtext(img_np, detail=1, paragraph=False)

        if not results:
            return "", 0.0

        lines = [text for _, text, _ in results]
        confidences = [conf for _, _, conf in results]
        mean_conf = sum(confidences) / len(confidences)

        raw_text = "\n".join(lines)
        logger.info(
            f"OCR complete: {len(lines)} lines, "
            f"mean confidence={mean_conf:.2f}"
        )
        return raw_text, mean_conf

    def _clean_text(self, text: str) -> str:
        text = text.replace("\u00b5", "u").replace("\u03bc", "u")
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        text = re.sub(r"(?<=\d)O(?=\d)", "0", text)
        text = re.sub(r"\bl(?=\d)", "1", text)

        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_lab_parameters(self, text: str) -> StructuredLabResult | None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        parameters: dict[str, LabParameter] = {}
        warnings: list[str] = []

        for line in lines:
            param_name = self._match_parameter(line)
            if not param_name:
                continue

            match = VALUE_UNIT_RE.search(line)
            if not match:
                continue

            raw_value_str = match.group("value")
            raw_value_str = raw_value_str.replace(",", ".")

            try:
                value = float(raw_value_str)
            except ValueError:
                continue

            if math.isnan(value) or math.isinf(value):
                continue

            raw_unit = match.group("unit") or ""
            unit = self._normalise_unit(raw_unit)

            valid, warning = self._validate_value(param_name, value, unit)
            if not valid:
                warnings.append(warning)
                continue

            flag = self._compute_flag(param_name, value)

            nr = NORMAL_RANGES.get(param_name)
            normal_range_str = f"{nr[0]}–{nr[1]} {nr[2]}" if nr else None

            parameters[param_name] = LabParameter(
                value=value,
                unit=unit or (nr[2] if nr else ""),
                raw_value=f"{match.group('modifier') or ''}{raw_value_str} {raw_unit}".strip(),
                normal_range=normal_range_str,
                flag=flag,
            )

        if not parameters:
            logger.warning("No lab parameters extracted from OCR text.")
            return None

        lab_name = self._extract_lab_name(text)
        date = self._extract_date(text)
        patient = self._extract_patient_name(text)

        return StructuredLabResult(
            parameters=parameters,
            raw_text=text,
            confidence=1.0,
            lab_name=lab_name,
            patient_name=patient,
            date=date,
            warnings=warnings,
        )

    def _match_parameter(self, line: str) -> str | None:
        lower = line.lower()

        for alias, canonical in PARAM_ALIASES.items():
            if alias in lower:
                return canonical

        tokens = re.findall(r"[a-z0-9\-/]+", lower)
        for token in tokens:
            if len(token) < 3:
                continue
            for alias, canonical in PARAM_ALIASES.items():
                if abs(len(token) - len(alias)) <= 1:
                    dist = self._edit_distance(token, alias)
                    if dist <= 1:
                        return canonical

        return None

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

    @staticmethod
    def _normalise_unit(unit: str) -> str:
        return UNIT_MAP.get(unit, unit.lower().strip())

    @staticmethod
    def _validate_value(param: str, value: float, unit: str) -> tuple[bool, str]:
        hard_limits: dict[str, tuple[float, float]] = {
            "hemoglobin":     (1.0, 25.0),
            "hematocrit":     (5.0, 70.0),
            "glucose":        (10, 2000),
            "fasting_blood_sugar": (10, 2000),
            "hba1c":          (3.0, 20.0),
            "creatinine":     (0.1, 50.0),
            "cholesterol_total": (50, 1000),
            "platelet_count": (1, 3000),
            "wbc_count":      (0.1, 500),
            "tsh":            (0.001, 200),
            "sodium":         (100, 180),
            "potassium":      (1.0, 10.0),
        }
        limits = hard_limits.get(param)
        if limits and not (limits[0] <= value <= limits[1]):
            msg = (
                f"Value {value} for '{param}' is outside plausible range "
                f"{limits} — likely OCR error. Skipping."
            )
            return False, msg
        return True, ""

    @staticmethod
    def _compute_flag(param: str, value: float) -> Optional[str]:
        nr = NORMAL_RANGES.get(param)
        if not nr:
            return None
        low, high, _ = nr
        if value < low:
            return "L"
        if value > high:
            return "H"
        return "N"

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        patterns = [
            r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",
            r"\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b",
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_lab_name(text: str) -> Optional[str]:
        first_lines = text.splitlines()[:3]
        for line in first_lines:
            line = line.strip()
            if len(line) > 4 and not re.match(r"^\d", line):
                return line[:80]
        return None

    @staticmethod
    def _extract_patient_name(text: str) -> Optional[str]:
        m = re.search(
            r"(?:patient|name|patient name)\s*[:\-]?\s*([A-Za-z ,\.]{3,50})",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return None
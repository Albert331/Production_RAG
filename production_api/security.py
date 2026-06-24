import re
from typing import Optional
from langsmith import traceable
import time

class RequestTimer:
    def __init__(self):
        self.elapsed_ms = 0
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000

class InputSanitizer:
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous\s+)?instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"you\s+are\s+now",
        r"you\s+have\s+no\s+restrictions",
    ]

    def __init__(self):
        self.patterns = [
            re.compile(p,re.IGNORECASE)
            for p in self.INJECTION_PATTERNS
        ]

    def detect(self,text: str) -> tuple[bool,Optional[str]]:
        for pattern in self.patterns:
            if pattern.search(text):
                return False, f"Suspicious pattern detected: {pattern.pattern}"
        return True,None
    
    def sanitize(self, text: str) -> str:
        """Remove potentially dangerous content."""

        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")

        return text.strip()


    
class PIIDetector:
    """Detect and mask personally identifiable information."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text."""
        found = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches
        return found

    def mask(self, text: str) -> str:
        """Mask PII in text."""
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            if pii_type == "email":
                masked = re.sub(pattern, "[EMAIL REDACTED]", masked)
            elif pii_type == "phone":
                masked = re.sub(pattern, "[PHONE REDACTED]", masked)
            elif pii_type == "ssn":
                masked = re.sub(pattern, "[SSN REDACTED]", masked)
            elif pii_type == "credit_card":
                masked = re.sub(pattern, "[CARD REDACTED]", masked)
            elif pii_type == "ip_address":
                masked = re.sub(pattern, "[IP REDACTED]", masked)
        return masked

class OutputValidator:
    """Validate LLM outputs before returning to user."""

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[bool, str, Optional[str]]:
        """
        Validate output.
        Returns: (is_valid, cleaned_output, reason_if_invalid)
        """
        # Check for PII leakage
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            cleaned = self.pii_detector.mask(output)
            return False, cleaned, f"PII detected and masked: {list(pii_found.keys())}"

        # Check for harmful content patterns
        harmful_patterns = [
            r"here('s| is) (how|the way) to (hack|steal|attack)",
            r"password is",
            r"api[_\s]?key",
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return (
                    False,
                    "[CONTENT BLOCKED]",
                    "Potentially harmful content detected",
                )

        return True, output, None

class SecurePipeline:
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable
    def check_input(self,text:str) -> tuple[bool,str,list[str]]:  
        notes=[]
        is_safe,reason = self.sanitizer.detect(text)
        if not is_safe:
            return False,'',[reason]
        

        cleaned = self.sanitizer.sanitize(text)

        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f'input PII masked: {list(pii_found.keys())}')

        return True, cleaned, notes    


    @traceable(name='security_check_output')
    def check_output(self,text:str)->tuple[str,list[str]]:
        return self.output_validator.validate(text)
    

    



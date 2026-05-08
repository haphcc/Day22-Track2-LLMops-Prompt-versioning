import re
import json
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    def validate(self, value, metadata):
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ssn_pattern  = r'\d{3}-\d{2}-\d{4}'
        found = []
        if re.search(email_pattern, value): found.append("email")
        if re.search(phone_pattern, value): found.append("phone")
        if re.search(ssn_pattern, value): found.append("SSN")
        if found:
            redacted = value
            redacted = re.sub(email_pattern, "[EMAIL_REDACTED]", redacted)
            redacted = re.sub(phone_pattern, "[PHONE_REDACTED]", redacted)
            redacted = re.sub(ssn_pattern, "[SSN_REDACTED]", redacted)
            return FailResult(error_message=f"PII detected: {', '.join(found)}", fix_value=redacted)
        return PassResult()

@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    def validate(self, value, metadata):
        try:
            json.loads(value)
            return PassResult()
        except:
            try:
                repaired = re.sub(r'```json\s*|\s*```', '', value).strip()
                repaired = repaired.replace("'", '"')
                json.loads(repaired)
                return FailResult(error_message="Invalid JSON format", fix_value=repaired)
            except:
                error_json = json.dumps({"error": "Failed to repair JSON", "raw": value})
                return FailResult(error_message="Fatal JSON error", fix_value=error_json)

def main():
    print("=" * 60)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 60)
    pii_guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    pii_cases = ["Contact me at john.doe@example.com", "My phone is 555-0199.", "Clean string here."]
    for case in pii_cases:
        result = pii_guard.validate(case)
        print(f"PII Test -> Input: {case[:40]} -> Output: {result.validated_output}")
    json_guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    json_cases = ['{"status": "ok"}', '```json\n{"status": "ok"}\n```', "{'status': 'ok'}"]
    for case in json_cases:
        result = json_guard.validate(case)
        print(f"JSON Test -> Input: {case[:40]} -> Output: {result.validated_output}")

if __name__ == "__main__":
    main()

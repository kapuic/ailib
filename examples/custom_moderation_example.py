"""Example showing how to integrate custom moderation services with AILib."""

from collections.abc import Callable
from typing import Any

from ailib import CompletionResponse
from ailib.safety import add_custom_filter


# Example 1: Perspective API Integration
def create_perspective_moderation(api_key: str) -> tuple[Callable, Callable]:
    """Create moderation hooks using Google's Perspective API.

    Perspective API analyzes text for toxicity, threats, etc.
    """

    def pre_hook(prompt: str, **kwargs) -> str:
        # Mock implementation - in reality, you'd call Perspective API
        # import requests
        # response = requests.post(
        #     "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze",
        #     params={"key": api_key},
        #     json={
        #         "comment": {"text": prompt},
        #         "requestedAttributes": {"TOXICITY": {}}
        #     }
        # )
        # score = response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]  # noqa: E501
        # if score > 0.7:
        #     raise ValueError("Content flagged as toxic")
        return prompt

    def post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
        # Similar check for response
        return response

    return pre_hook, post_hook


# Example 2: Azure Content Moderator Integration
def create_azure_moderation(endpoint: str, api_key: str) -> tuple[Callable, Callable]:
    """Create moderation hooks using Azure Content Moderator.

    Azure provides text moderation for profanity, PII, etc.
    """

    def check_with_azure(text: str) -> dict[str, Any]:
        # Mock implementation
        # import requests
        # headers = {
        #     "Content-Type": "text/plain",
        #     "Ocp-Apim-Subscription-Key": api_key
        # }
        # response = requests.post(
        #     f"{endpoint}/contentmoderator/moderate/v1.0/ProcessText/Screen",
        #     headers=headers,
        #     data=text
        # )
        # return response.json()
        return {"Classification": {"ReviewRecommended": False}}

    def pre_hook(prompt: str, **kwargs) -> str:
        result = check_with_azure(prompt)
        if result.get("Classification", {}).get("ReviewRecommended"):
            raise ValueError("Content requires review")
        return prompt

    def post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
        result = check_with_azure(response.content)
        if result.get("Classification", {}).get("ReviewRecommended"):
            response.content = "Content requires review before display"
        return response

    return pre_hook, post_hook


# Example 3: Local ML Model Moderation
def create_local_ml_moderation(model_path: str) -> tuple[Callable, Callable]:
    """Create moderation using a local ML model (e.g., Hugging Face).

    This example shows how to use transformers for local moderation.
    """
    # Mock implementation - in reality:
    # from transformers import pipeline
    # classifier = pipeline("text-classification", model=model_path)

    def classify_text(text: str) -> dict[str, float]:
        # Mock classification
        # results = classifier(text)
        # return {r["label"]: r["score"] for r in results}
        return {"SAFE": 0.95, "UNSAFE": 0.05}

    def pre_hook(prompt: str, **kwargs) -> str:
        scores = classify_text(prompt)
        if scores.get("UNSAFE", 0) > 0.5:
            raise ValueError("Content classified as unsafe")
        return prompt

    def post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
        scores = classify_text(response.content)
        if scores.get("UNSAFE", 0) > 0.5:
            response.content = "Content filtered by ML model"
        return response

    return pre_hook, post_hook


# Example 4: Custom Business Logic Moderation
def create_business_rules_moderation(
    forbidden_terms: list[str], sensitive_data_patterns: list[str]
) -> tuple[Callable, Callable]:
    """Create moderation based on custom business rules."""

    import re

    def check_business_rules(text: str) -> list[str]:
        violations = []

        # Check forbidden terms
        text_lower = text.lower()
        for term in forbidden_terms:
            if term.lower() in text_lower:
                violations.append(f"Forbidden term: {term}")

        # Check sensitive data patterns
        for pattern in sensitive_data_patterns:
            if re.search(pattern, text):
                violations.append("Sensitive data pattern detected")

        return violations

    def pre_hook(prompt: str, **kwargs) -> str:
        violations = check_business_rules(prompt)
        if violations:
            raise ValueError(f"Business rule violations: {', '.join(violations)}")
        return prompt

    def post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
        violations = check_business_rules(response.content)
        if violations:
            response.content = "Content blocked due to policy violations"
        return response

    return pre_hook, post_hook


# Example 5: Combining Multiple Moderation Services
class CombinedModerator:
    """Combine multiple moderation services with fallback."""

    def __init__(self):
        self.moderators = []

    def add_moderator(
        self, name: str, pre_hook: Callable, post_hook: Callable, required: bool = True
    ):
        """Add a moderation service."""
        self.moderators.append(
            {
                "name": name,
                "pre_hook": pre_hook,
                "post_hook": post_hook,
                "required": required,
            }
        )

    def create_hooks(self) -> tuple[Callable, Callable]:
        """Create combined hooks."""

        def combined_pre_hook(prompt: str, **kwargs) -> str:
            for mod in self.moderators:
                try:
                    prompt = mod["pre_hook"](prompt, **kwargs)
                except Exception as e:
                    if mod["required"]:
                        raise
                    print(f"Warning: {mod['name']} pre-check failed: {e}")
            return prompt

        def combined_post_hook(
            response: CompletionResponse, **kwargs
        ) -> CompletionResponse:
            for mod in self.moderators:
                try:
                    response = mod["post_hook"](response, **kwargs)
                except Exception as e:
                    if mod["required"]:
                        raise
                    print(f"Warning: {mod['name']} post-check failed: {e}")
            return response

        return combined_pre_hook, combined_post_hook


# Example Usage
def demo_custom_moderation():
    """Demonstrate custom moderation integration."""
    print("Custom Moderation Examples")
    print("=" * 50)

    # 1. Using Perspective API
    print("\n1. Perspective API Integration:")
    perspective_pre, perspective_post = create_perspective_moderation("your-api-key")
    print("✓ Created Perspective API hooks")

    # 2. Using Azure Content Moderator
    print("\n2. Azure Content Moderator:")
    azure_pre, azure_post = create_azure_moderation(
        "https://your-endpoint.cognitiveservices.azure.com", "your-api-key"
    )
    print("✓ Created Azure moderation hooks")

    # 3. Using Local ML Model
    print("\n3. Local ML Model:")
    ml_pre, ml_post = create_local_ml_moderation("path/to/model")
    print("✓ Created local ML moderation hooks")

    # 4. Using Business Rules
    print("\n4. Business Rules:")
    business_pre, business_post = create_business_rules_moderation(
        forbidden_terms=["competitor", "confidential"],
        sensitive_data_patterns=[
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
        ],
    )
    print("✓ Created business rules hooks")

    # 5. Combining Multiple Services
    print("\n5. Combined Moderation:")
    combined = CombinedModerator()

    # Add OpenAI as primary (required)
    from ailib.safety import with_moderation

    try:
        openai_pre, openai_post = with_moderation()
        combined.add_moderator("OpenAI", openai_pre, openai_post, required=True)
    except Exception:
        print("  - OpenAI moderation unavailable (no API key)")

    # Add Perspective as secondary (optional)
    combined.add_moderator(
        "Perspective", perspective_pre, perspective_post, required=False
    )

    # Add business rules as required
    combined.add_moderator("Business Rules", business_pre, business_post, required=True)

    # Create final hooks
    final_pre, final_post = combined.create_hooks()
    print("✓ Created combined moderation with fallback")

    # Use with AILib
    print("\n6. Integration with AILib:")
    print("Pass hooks to agents or use globally:")
    print(
        """
    # Option 1: Use with specific agent
    agent = Agent(
        llm=client,
        pre_hook=final_pre,
        post_hook=final_post
    )

    # Option 2: Add as custom filter
    add_custom_filter(lambda text: business_rules_check(text))

    # Option 3: Extend safety configuration
    enable_safety(
        custom_filters=[...],
        rate_limit=60
    )
    """
    )


# Example: Simple Custom Filter
def demo_simple_integration():
    """Show the simplest way to add custom moderation."""
    print("\nSimplest Custom Moderation:")
    print("=" * 50)

    # Method 1: Lambda filter
    add_custom_filter(lambda text: "banned_word" in text.lower())
    print("✓ Added simple word filter")

    # Method 2: Function filter
    def check_company_policy(text: str) -> bool:
        """Return True if text violates policy."""
        # Your custom logic here
        if "internal only" in text.lower():
            return True
        if len(text) > 10000:
            return True
        return False

    add_custom_filter(check_company_policy)
    print("✓ Added company policy filter")

    # Method 3: External API filter
    def check_external_api(text: str) -> bool:
        """Check with external moderation API."""
        # Mock API call
        # response = requests.post("https://api.example.com/moderate", json={"text": text})  # noqa: E501
        # return response.json()["blocked"]
        return False

    add_custom_filter(check_external_api)
    print("✓ Added external API filter")


if __name__ == "__main__":
    demo_custom_moderation()
    print("\n" + "=" * 50)
    demo_simple_integration()

    print("\n✅ Custom moderation examples completed!")
    print("\nKey Takeaways:")
    print("- AILib's moderation is pluggable and extensible")
    print("- OpenAI moderation is optional, not required")
    print("- Easy to integrate any moderation service")
    print("- Can combine multiple services with fallback")
    print("- Simple filters can be added with one line")

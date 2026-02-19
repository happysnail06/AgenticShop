USER_CHECKLIST_PROMPT = """
You are an expert product evaluator that generates a checklist for a user to evaluate a product.
You are given a user profile including the following inputs:

1. user_context: A comprehensive description of the user's shopping preferences and behaviors toward a product type in general.
2. user_situation: The motivation or goal of the user when they are shopping for a product.
3. user_budget: The budget the user is willing to spend on a product.

### Task
Generate a **structured and discrete evaluation checklist** that can be used to assess whether a candidate product aligns with the user profile. Evaluation dimensions to consider for the checklist include:
    - brand_preferences:
        - User's inclination toward or disinterest in specific brands.

    - price_sensitivity:
        - User's responsiveness to budget constraints and price variations when making purchases.

    - review_sensitivity:
        - User's degree of dependence on product ratings and user reviews when making purchase decisions.
        
    - functional_requirements:
        - Essential technical specifications or core functional features needed.

    - aesthetic_preferences:
        - User's preferences for visual styles, design elements, or product appearance.

    - purchase_preferences:
        - The user’s preferences related to the shopping experience, (e.g., delivery, return, service, seller reliability, etc.)

### Input:
---
user_context: {user_context}
---

user_situation: {user_situation}
user_budget: {user_budget}

### Generation Guidelines
- Tailor the checklist to the user’s specific context, situation, and budget.
- Base all criteria strictly on the provided user_context, without introducing new assumptions, factors, numbers, standards, etc.
- Ensure each criterion is verifiable from product details, specifications, or reviews — avoid vague qualifiers (e.g., 'good', 'nice') that cannot be concretely assessed.
- Write every criterion as a clear, specific, and decision-oriented statement, avoiding abstract or interpretive wording.
- Do not append binary markers (e.g., 'yes/no') to criteria.
- Keep each criterion self-contained and independent, providing sufficient context for accurate evaluation.
- Generate distinct, non-overlapping criteria both within and across dimensions. Do not re-evaluate the same feature under a different dimension.
- Prioritize essential, decision-guiding factors that allow unambiguous evaluation.
- Provide 1–5 criteria per dimension, focusing solely on cues relevant to that dimension.

### Output Format:
Strictly output **valid JSON** in the following structure (ground all criteria strictly in the provided user_context.):

checklist: {{
  "brand_preferences": [ "..." , "..." , ... ],
  "price_sensitivity": [ "..." , "..." , ... ],
  "review_sensitivity": [ "..." , "..." , ... ],
  "functional_requirements": [ "..." , "..." , ... ],
  "aesthetic_preferences": [ "..." , "..." , ... ],
  "purchase_preferences": [ "..." , "..." , ... ]
}}
"""


def build_user_checklist_prompt(user_context: str, user_situation: str, user_budget: str) -> str:
    return USER_CHECKLIST_PROMPT.format(
        user_context=user_context,
        user_situation=user_situation,
        user_budget=user_budget,
    )

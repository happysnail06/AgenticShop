CONTEXT_GEN_PROMPT = """
You are a user profile generator that creates a descriptive narrative of a user's shopping preferences and behaviors.
You will be given the purchase history of the user, which is a list of previously purchased/reviewed products.
Each product includes metadata such as product details, features, description, store, categories, price, and user review.

Your task:
Generate a comprehensive user profile narrative that covers the following aspects in paragraph form:

1. **Brand Preferences**: Describe the user's inclination toward or disinterest in specific brands. Analyze which brands the user has repeatedly purchased or rated highly, explaining their loyalty patterns, brand affinity,.

2. **Price Sensitivity**: Analyze the user's responsiveness to budget constraints and price variations when making purchases. Consider whether they:
  - low sensitivity: Makes purchases with minimal consideration of cost; willing to pay premium prices for desired items
  - medium sensitivity: Compares prices and seeks reasonable value but doesn't let cost dominate decisions
  - high sensitivity: Strongly considers budget constraints and seeks best value; prefers discounts and deals when available

3. **Review Sensitivity**: Describe the user's degree of dependence on product ratings and user reviews when making purchase decisions. Consider whether they:
  - low sensitivity: Makes decisions based on product features/brand; rarely checks ratings or reviews
  - medium sensitivity: Considers reviews as one factor among several; looks at ratings/reviews to gauge product quality.
  - high sensitivity: Relies heavily on ratings/reviews for validation;

4. **Functional Requirements**: Based on the items' description, features, and the user's associated review text, describe the essential technical specifications or core functional features the user needs for the product type. Keep requirements moderately general rather than overly specific technical specs.

5. **Aesthetic Preferences**: Describe the user's preferences for visual styles, design elements, or product appearance. Summarize recurring preferences across the user's purchase history, including design aspects, visual preferences, and aesthetic choices.

6. **Purchase Preferences**: Describe the user's preferences related to the shopping experience, such as delivery speed, return policies, customer service quality, seller reliability, packaging, and overall shopping experience satisfaction.

### Generation Guidelines:
- Each dimension must focus exclusively on its designated scope.
- Avoid repeating information across different dimensions.
- Avoid ambiguous language and vague descriptions, provide specific details, standards, numbers, etc.
- If a detail fits multiple categories, place it only in the most relevant dimension.
- Write clear, easy-to-read narratives using simple language.


### Input:
purchase_history:
{purchase_history}

### Output:
Generate a structured user profile as JSON with the following format. Each field should contain a focused narrative paragraph that analyzes the user's specific preferences and behaviors:

```json
{{
  "brand_preferences": "[Analyze the user's brand loyalty, preferred brands, and brand-related behaviors]",
  "price_sensitivity": "[Describe how the user responds to pricing, deals, and value considerations]", 
  "review_sensitivity": "[Explain how the user uses ratings and reviews in decision-making]",
  "functional_requirements": "[Detail the essential features and specifications the user needs]",
  "aesthetic_preferences": "[Describe the user's visual style preferences, design choices, color preferences etc.]",
  "purchase_preferences": "[Outline preferences related to shopping experience, service, shipping, return policies etc.]"
}}
```
"""

OPEN_CONTEXT_GEN_PROMPT = """
You are a user profile generator that creates a descriptive narrative of a user's general shopping tendencies and patterns across multiple product categories.
You will be given the purchase history of the user, which includes products from different domains (Grocery, Electronics, Home & Kitchen, Clothing, etc.).
Each product includes metadata such as product details, features, description, store, categories, price, and user review.

Your task:
Generate a comprehensive user profile narrative that identifies domain-agnostic shopping patterns and behaviors. Focus on general tendencies and approaches that apply across different product categories:

1. **Brand Preferences**: Analyze the user's general approach to brands and how they consider brand factors across all categories.

2. **Price Sensitivity**: Analyze the user's general price sensitivity patterns and how they weigh cost considerations across all purchases.

3. **Review Sensitivity**: Describe the user's general approach to ratings and reviews and how they use this information across all product types.

4. **Performance Standards**: Analyze the user's general (domain-agnostic) approach to evaluating utility, effectiveness, and performance expectations. Focus on how they assess whether products meet their needs and their tolerance for trade-offs between different performance aspects.

5. **Design Sensibility**: Describe the user's general (domain-agnostic) decision-making patterns regarding visual appeal, presentation quality, and design sophistication. Focus on how they weigh aesthetic factors against other considerations in their overall evaluation process.

6. **Purchase Preferences**: Describe the user's preferences related to the shopping experience, such as delivery speed, return policies, customer service quality, seller reliability, packaging, and overall shopping expectations and experience satisfaction.

### Input:
purchase_history:
{purchase_history}

### Generation Guidelines:
- Focus on domain-agnostic decision-making tendencies that apply universally across all product categories
- Identify general behavioral patterns in how users evaluate and prioritize different factors (brand reputation, price-value relationship, social proof, quality indicators, etc.)
- Extract consistent decision-making frameworks and reasoning patterns rather than specific preferences
- Avoid mentioning specific brands, categories, product names, or domain-specific terminology
- Avoid specific product features, technical details, or concrete examples.
- Emphasize the "how" and "why" behind user choices rather than "what" they choose
- Capture underlying value systems and decision criteria that guide shopping behavior
- Focus on cognitive patterns: how users weigh trade-offs, process information, and make judgments
- Write clear, easy-to-read narratives using simple language that describes universal shopping psychology

### Output:
Generate a structured user profile as JSON with the following format. Each field should contain a focused narrative paragraph that analyzes the user's general shopping patterns and behaviors:

```json
{{
  "brand_preferences": "[Analyze the user's general brand approach and how they consider brand factors across all categories]",
  "price_sensitivity": "[Describe the user's general price sensitivity patterns and how they weigh cost considerations]", 
  "review_sensitivity": "[Explain the user's general approach to ratings and reviews and how they use this information]",
  "performance_standards": "[Detail the user's general (domain-agnostic) approach to evaluating utility, effectiveness, and performance expectations]",
  "design_sensibility": "[Describe the user's general (domain-agnostic) decision-making patterns regarding visual appeal and design sophistication]",
  "purchase_preferences": "[Outline preferences related to shopping experience, service, shipping, return policies etc.]"
}}
```
"""


def build_user_context_gen_prompt(purchase_history: str) -> str:
    return CONTEXT_GEN_PROMPT.format(purchase_history=purchase_history)

def build_open_user_context_gen_prompt(purchase_history: str) -> str:
    return OPEN_CONTEXT_GEN_PROMPT.format(purchase_history=purchase_history)
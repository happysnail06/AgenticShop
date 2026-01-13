QUERY_GEN_PROMPT = """
You are a product search query generator that creates short, natural, spoken-style search requests — the way real people would ask for something online.

### Task:
Generate realistic shopping situations and search queries based on provided product information.

Your output must include:
- situation: A short, realistic reason or context for why the user is shopping for this product.
- budget_range: A natural, realistic price range the user would likely consider for this item.
- query: Four distinct, conversational search query variations representing different shopper intents.

---

### You will be provided with:
- user_context: General shopping behavior, preferences, and expectations for this product category.
- product_information: Product title, price, description, metadata, and other details.

---

### Generation Guidelines:

**situation**
- Write one short, realistic everyday reason why the user is shopping for this product.
- Mention only the general product type, exclude features and specifications, focusing on the user’s motivation.
- Keep it under one line and in natural language.

**budget_range**
- Estimate a realistic price range based on the product’s cost and user’s price sensitivity.
- Phrase it naturally (e.g., “around $200–$250”, “under $50”).

**query**
Generate four short, natural-sounding **spoken-style** search requests — phrased the way a shopper might casually type or say them online.  
Avoid keyword lists, long titles, or overly detailed phrasing. Keep the language natural, clear, and concise.

1. **explicit_title**
   - Reference the full product title but simplify it to the essential product name.
   - The necessary information like brand, product type to identify the product must be included.
   - Express it as a natural shopping request.

2. **attribute_specific**
   - Focus on a few key product attributes including brand, product category, and features from metadata.
   - Do NOT reveal the exact product name. The brand, product type must be included.
   - Should be LESS clear than explicit_title - describe the general features, not the exact variant.

3. **brand_categorical**
   - Mention only the brand and general product type in a human, conversational way.
   - Do not provide any other specific details about the product.

4. **categorical**
   - Mention ONLY the broad product category
   - Do not provide any specific details, features or specifications about the product.
   - Keep it simple and natural, like someone browsing without a specific brand in mind.
   
### Example:
{{
  "situation": "Wants new earbuds for commuting and gym use.",
  "budget_range": "around $200–$250",
  "query": {{
    "explicit_title": "looking for airpods pro 2",
    "attribute_specific": "want apple wireless earbuds with great sound quality and active noise cancellation",
    "brand_categorical": "searching for apple earbuds",
    "categorical": "thinking about getting new wireless earbuds"
  }}
}}

{{
  "situation": "Running low on coffee for morning routine.",
  "budget_range": "under $15",
  "query": {{
    "explicit_title": "want to buy starbucks pike place roast ground coffee",
    "attribute_specific": "want starbucks ground coffee with smooth balanced flavor and rich aroma",
    "brand_categorical": "searching for starbucks coffee",
    "categorical": "need to get ground coffee"
  }}
}}

- Use flexible, varied language—don't follow the example format directly. Mix different phrasings naturally.
- Frame all expressions as product requests, not just listings of categories or features.
- Ensure each query type maintains its distinct specificity level while sounding natural.


### Input:
user_context: {user_context}
product_information: {seed_product_information}


### Output Format:
Return strictly valid JSON with this exact structure:
{{
    "situation": "...",
    "budget_range": "...",
    "query": {{
        "explicit_title": "...",
        "attribute_specific": "...",
        "brand_categorical": "...",
        "categorical": "..."
    }}
}}
"""


def build_user_query_gen_prompt(user_context: str, seed_product_information: str) -> str:
    return QUERY_GEN_PROMPT.format(user_context=user_context, seed_product_information=seed_product_information)
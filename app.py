import os
import json
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# 1) LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# =========================================================
# 2) CREATE GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# 3) LOAD CONFIG
# =========================================================

def load_config():
    with open(
        "config.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# =========================================================
# 4) GET USER TICKET
# =========================================================

def get_user_ticket():
    ticket = input(
        "\nEnter customer ticket:\n> "
    )

    return ticket


# =========================================================
# 5) ANALYZE TICKET WITH LLM
# =========================================================

def analyze_ticket(ticket, config):

    categories = json.dumps(
        config["categories"],
        indent=2
    )

    priority_rules = json.dumps(
        config["priority_rules"],
        indent=2
    )

    output_format = json.dumps(
        config["analysis_output"],
        indent=2
    )

    user_prompt = f"""
TASK:
Analyze the following customer support ticket.

CUSTOMER TICKET:
<ticket>
{ticket}
</ticket>

ALLOWED CATEGORIES:
{categories}

PRIORITY RULES:
{priority_rules}

INSTRUCTIONS:
1. Determine the sentiment.
2. Choose one category from the allowed categories.
3. Determine the customer's main intent.
4. Determine the priority.
5. Write a short summary.

CONSTRAINTS:
- Use only information available in the ticket.
- Do not invent missing information.
- Category must be one of the allowed categories.
- Priority must be low, medium, or high.
- Return only valid JSON.
- Do not add markdown.

OUTPUT FORMAT:
{output_format}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": config["system_prompt"]
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,
        max_tokens=300
    )

    model_output = (
        response
        .choices[0]
        .message
        .content
    )

    try:
        analysis = json.loads(model_output)

    except json.JSONDecodeError:
        print("\nERROR: Invalid JSON from model.")
        print(model_output)
        raise

    return analysis


# =========================================================
# 6) VALIDATE LLM OUTPUT
# =========================================================

def validate_analysis(analysis, config):

    required_fields = [
        "sentiment",
        "category",
        "intent",
        "priority",
        "summary"
    ]

    for field in required_fields:
        if field not in analysis:
            raise ValueError(
                f"Missing field: {field}"
            )

    allowed_categories = config["categories"]

    if analysis["category"] not in allowed_categories:
        raise ValueError(
            f"Invalid category: {analysis['category']}"
        )

    allowed_priorities = [
        "low",
        "medium",
        "high"
    ]

    if analysis["priority"] not in allowed_priorities:
        raise ValueError(
            f"Invalid priority: {analysis['priority']}"
        )

    return True


# =========================================================
# 7) DECIDE BUSINESS ACTION
# =========================================================

def decide_action(analysis):

    category = analysis["category"]
    priority = analysis["priority"]

    if priority == "high":
        action = "Escalate immediately"

    elif priority == "medium":
        action = "Send to support queue"

    else:
        action = "Normal support handling"


    if category == "billing":
        team = "Billing Team"

    elif category == "technical":
        team = "Technical Support"

    elif category == "account":
        team = "Account Support"

    elif category == "delivery":
        team = "Delivery Team"

    elif category == "refund":
        team = "Refund Team"

    else:
        team = "General Support"

    return {
        "action": action,
        "assigned_team": team
    }


# =========================================================
# 8) GENERATE SUGGESTED CUSTOMER REPLY
# =========================================================

def generate_reply(ticket, analysis):

    prompt = f"""
You are a professional customer support agent.

Write a short and professional reply to the customer.

CUSTOMER TICKET:
<ticket>
{ticket}
</ticket>

TICKET ANALYSIS:
{json.dumps(analysis, indent=2)}

RULES:
- Be polite.
- Acknowledge the customer's issue.
- Do not promise actions that are not confirmed.
- Keep the response under 100 words.
- Return only the reply text.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.4,
        max_tokens=150
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# 9) SAVE TICKET
# =========================================================

def save_ticket(
    ticket,
    analysis,
    business_action,
    suggested_reply
):

    os.makedirs(
        "tickets",
        exist_ok=True
    )

    ticket_record = {
        "created_at": datetime.now().isoformat(),
        "original_ticket": ticket,
        "analysis": analysis,
        "business_action": business_action,
        "suggested_reply": suggested_reply
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"tickets/ticket_{timestamp}.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            ticket_record,
            file,
            indent=2,
            ensure_ascii=False
        )

    return filename


# =========================================================
# 10) MAIN PROGRAM
# =========================================================

def main():

    config = load_config()

    ticket = get_user_ticket()

    print("\nAnalyzing ticket...")

    analysis = analyze_ticket(
        ticket,
        config
    )

    validate_analysis(
        analysis,
        config
    )

    business_action = decide_action(
        analysis
    )

    print("\nGenerating suggested reply...")

    suggested_reply = generate_reply(
        ticket,
        analysis
    )

    saved_file = save_ticket(
        ticket,
        analysis,
        business_action,
        suggested_reply
    )


    print("\n" + "=" * 60)
    print("TICKET ANALYSIS")
    print("=" * 60)

    print(
        "Sentiment:",
        analysis["sentiment"]
    )

    print(
        "Category:",
        analysis["category"]
    )

    print(
        "Intent:",
        analysis["intent"]
    )

    print(
        "Priority:",
        analysis["priority"]
    )

    print(
        "Summary:",
        analysis["summary"]
    )


    print("\n" + "=" * 60)
    print("BUSINESS ACTION")
    print("=" * 60)

    print(
        "Action:",
        business_action["action"]
    )

    print(
        "Assigned Team:",
        business_action["assigned_team"]
    )


    print("\n" + "=" * 60)
    print("SUGGESTED REPLY")
    print("=" * 60)

    print(
        suggested_reply
    )


    print(
        "\nSaved to:",
        saved_file
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()
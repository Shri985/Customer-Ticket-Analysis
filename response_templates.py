"""
response_templates.py
Professional auto-response templates keyed by (category, priority).
"""

RESPONSES = {
    ("Billing & Payments", "High"): """Dear Valued Customer,

Thank you for reaching out to us. We understand this is an urgent billing matter and sincerely apologize for the inconvenience caused.

Your case has been escalated to our Priority Billing Team and will be reviewed within the next 2 hours. Our team will:
• Investigate the transaction discrepancy immediately
• Initiate a refund or correction if applicable
• Provide you with a full resolution update via email

Reference ID: {ticket_id} | Priority: HIGH

We appreciate your patience and assure you this will be resolved promptly.

Best regards,
Customer Support Team""",

    ("Billing & Payments", "Medium"): """Dear Valued Customer,

Thank you for contacting our Billing Support team.

We have received your query regarding a billing concern and have assigned it to our finance team for review. You can expect a detailed response within 24 hours.

To expedite the process, please have the following ready:
• Transaction ID or Invoice Number
• Date of the transaction
• Amount in question

Reference ID: {ticket_id}

We appreciate your patience.

Best regards,
Billing Support Team""",

    ("Billing & Payments", "Low"): """Dear Valued Customer,

Thank you for getting in touch with us.

We have logged your billing inquiry and our team will review it within 2–3 business days. You will receive a detailed response once the review is complete.

Reference ID: {ticket_id}

Thank you for your patience.

Best regards,
Customer Support Team""",

    ("Technical Support", "High"): """Dear Valued Customer,

We sincerely apologize for the critical technical issue you are experiencing. This has been flagged as HIGH PRIORITY and escalated to our Senior Technical Team.

Immediate steps being taken:
• A senior engineer has been assigned to your case
• Remote diagnostics will begin within 30 minutes
• You will receive a status update within 1 hour

Reference ID: {ticket_id} | Priority: HIGH

As a temporary workaround, please try:
1. Clearing your browser cache and cookies
2. Restarting the application
3. Using an incognito/private window

We are committed to resolving this as quickly as possible.

Best regards,
Technical Support Team""",

    ("Technical Support", "Medium"): """Dear Valued Customer,

Thank you for reporting this technical issue. We have logged your ticket and our technical team will investigate it within 4–8 hours.

Reference ID: {ticket_id}

While we work on a fix, you may try:
• Refreshing the page or restarting the app
• Checking our status page at status.support.com
• Clearing your browser cache

We will keep you updated on the progress.

Best regards,
Technical Support Team""",

    ("Technical Support", "Low"): """Dear Valued Customer,

Thank you for reaching out to Technical Support.

We have received your report and our team will look into it within 1–2 business days. We will notify you once a resolution is available.

Reference ID: {ticket_id}

Best regards,
Technical Support Team""",

    ("Account Management", "High"): """Dear Valued Customer,

We take account security and management issues very seriously. Your request has been flagged as URGENT and our Account Security Team has been notified immediately.

Actions being taken:
• Your account has been flagged for priority review
• Security protocols have been initiated
• You will be contacted within 1 hour

Reference ID: {ticket_id} | Priority: HIGH

If you suspect unauthorized access, please change your password immediately and enable two-factor authentication.

Best regards,
Account Security Team""",

    ("Account Management", "Medium"): """Dear Valued Customer,

Thank you for contacting Account Management.

Your request has been received and will be processed within 24 hours. Our team will verify your identity and make the requested changes securely.

Reference ID: {ticket_id}

For security purposes, please do not share your password or OTP with anyone.

Best regards,
Account Management Team""",

    ("Account Management", "Low"): """Dear Valued Customer,

Thank you for your account management request.

We have received your query and will process it within 2–3 business days. You will receive a confirmation email once the changes are applied.

Reference ID: {ticket_id}

Best regards,
Account Management Team""",

    ("Shipping & Delivery", "High"): """Dear Valued Customer,

We sincerely apologize for the urgent delivery issue you are facing. This has been escalated to our Logistics Priority Team.

Immediate actions:
• A delivery investigation has been initiated with the courier
• Our logistics team will contact you within 2 hours
• If the package is confirmed lost, a replacement will be dispatched within 24 hours

Reference ID: {ticket_id} | Priority: HIGH

Please keep your tracking number handy: it will be needed for the courier investigation.

Best regards,
Shipping & Logistics Team""",

    ("Shipping & Delivery", "Medium"): """Dear Valued Customer,

Thank you for reaching out about your delivery concern.

We have raised a query with our logistics partner and will provide you with an update within 24–48 hours. Our team is tracking your shipment and will ensure it reaches you at the earliest.

Reference ID: {ticket_id}

Best regards,
Shipping Support Team""",

    ("Shipping & Delivery", "Low"): """Dear Valued Customer,

Thank you for contacting Shipping Support.

We have noted your delivery query and will follow up with our courier partner within 2–3 business days.

Reference ID: {ticket_id}

Best regards,
Shipping Support Team""",

    ("Product & Service", "High"): """Dear Valued Customer,

We are sorry to hear about the urgent issue with your product or service. This has been escalated to our Product Quality Team for immediate review.

Actions being taken:
• A product specialist has been assigned to your case
• A replacement or refund will be processed within 24 hours if applicable
• You will receive a follow-up call within 2 hours

Reference ID: {ticket_id} | Priority: HIGH

Best regards,
Product Support Team""",

    ("Product & Service", "Medium"): """Dear Valued Customer,

Thank you for reaching out about your product or service concern.

Our product team will review your case and respond within 24–48 hours with a resolution or next steps.

Reference ID: {ticket_id}

Best regards,
Product Support Team""",

    ("Product & Service", "Low"): """Dear Valued Customer,

Thank you for contacting Product Support.

We have received your query and will respond within 3–5 business days with the relevant information or resolution.

Reference ID: {ticket_id}

Best regards,
Product Support Team""",
}

DEFAULT_RESPONSE = """Dear Valued Customer,

Thank you for contacting our support team.

We have received your ticket and it has been assigned to the appropriate team for review. You will receive a detailed response within 24–48 hours.

Reference ID: {ticket_id}

Best regards,
Customer Support Team"""


def get_response(category: str, priority: str, ticket_id: str) -> str:
    template = RESPONSES.get((category, priority), DEFAULT_RESPONSE)
    return template.format(ticket_id=ticket_id)

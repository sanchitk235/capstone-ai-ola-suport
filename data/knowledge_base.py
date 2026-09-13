"""
data/knowledge_base.py - KB articles for ola support policies.
12 docs covering SLA, refudns, escalations etc for RAG retrieval.
"""

KB_DOCUMENTS: list[dict] = [
    {
        "doc_id": "KB-001",
        "topic": "ticket_priority_classification",
        "content": (
            "Ola classifies every incoming support ticket into one of four priority tiers: "
            "P1 (Critical), P2 (High), P3 (Medium), and P4 (Low). "
            "P1 tickets represent complete service outages, safety incidents, or data-security "
            "breaches that affect a large number of riders or drivers simultaneously and "
            "must be acknowledged by a support engineer within 15 minutes of creation. "
            "P2 tickets cover payment failures, account lockouts, and driver-app crashes "
            "that prevent a single user from completing a ride, with a required acknowledgment "
            "window of one hour. "
            "P3 tickets address standard billing queries, minor app glitches, and general "
            "product questions, while P4 covers feature requests and non-urgent feedback, "
            "both handled during normal business hours."
        ),
    },
    {
        "doc_id": "KB-002",
        "topic": "sla_by_severity",
        "content": (
            "Ola's Service Level Agreements define both a first-response time and a "
            "resolution time for each priority tier. "
            "P1 tickets must be acknowledged within 15 minutes and fully resolved within "
            "4 hours, with a mandatory status update posted to the Ola Status Page every "
            "30 minutes until the issue is closed. "
            "P2 tickets require a first response within 1 hour and resolution within 12 hours; "
            "customers receive an automated progress notification every 2 hours. "
            "P3 tickets are acknowledged within 4 hours and resolved within 3 business days, "
            "while P4 tickets are acknowledged within 1 business day and resolved within "
            "7 business days. "
            "Breach of any SLA target automatically escalates the ticket to the next support "
            "tier and triggers a managerial alert."
        ),
    },
    {
        "doc_id": "KB-003",
        "topic": "escalation_matrix",
        "content": (
            "Ola operates a three-tier support escalation structure to ensure that unresolved "
            "tickets reach appropriately skilled staff. "
            "Tier 1 agents handle first-contact resolution for routine queries such as general "
            "inquiries, minor billing clarifications, and password resets. "
            "Tier 2 is staffed by senior agents with direct access to payment gateways and "
            "driver-management systems; any Tier 1 ticket that remains unresolved for more "
            "than 24 hours, or any ticket flagged escalated=True in the CRM, is automatically "
            "routed to Tier 2 within 30 minutes. "
            "Tier 3 involves specialist engineers and senior management and is reserved for "
            "P1/P2 tickets that breach their SLA, for issues with legal or regulatory "
            "implications, and for complaints from enterprise or VIP account holders that "
            "remain unsatisfied after Tier 2 intervention."
        ),
    },
    {
        "doc_id": "KB-004",
        "topic": "refund_compensation_policy",
        "content": (
            "Ola processes refunds for verified overcharges, ride cancellations caused by "
            "driver no-show or confirmed technical failures on the Ola platform. "
            "Approved refunds are returned to the original payment instrument within "
            "5–7 business days; wallet-based payments are refunded to the Ola wallet "
            "within 24 hours. "
            "When a ride is cancelled by Ola due to driver unavailability after the customer "
            "has been kept waiting for more than 10 minutes, the customer automatically "
            "receives a fare waiver for the cancelled booking and a service credit of up to "
            "INR 50 applied to their next ride. "
            "Compensation for delayed rides or ride-quality issues is assessed case-by-case "
            "by a Tier 2 agent and may include a partial fare refund or additional service "
            "credits, but will not exceed the original fare charged."
        ),
    },
    {
        "doc_id": "KB-005",
        "topic": "customer_communication_channel_policy",
        "content": (
            "Customers may contact Ola support through the in-app help centre, the Ola "
            "website live-chat widget, or the dedicated support email address; all three "
            "channels feed into the same CRM ticket thread so agents always have the full "
            "communication history. "
            "Phone support is reserved exclusively for P1 and P2 priority tickets and is "
            "available 24 hours a day, 7 days a week via a dedicated support line. "
            "All channel response-time commitments align with the SLA targets stated in the "
            "severity policy, regardless of which channel the customer used to raise the "
            "ticket. "
            "Customers are notified by push notification and email whenever their ticket "
            "status changes, ensuring transparency across channels."
        ),
    },
    {
        "doc_id": "KB-006",
        "topic": "business_hours_holiday_support",
        "content": (
            "Ola provides round-the-clock, 24/7 support coverage for P1 and P2 tickets "
            "throughout the calendar year, including all national public holidays and "
            "regional festive periods. "
            "P3 and P4 tickets are handled during standard business hours of 9:00 AM to "
            "9:00 PM IST, Monday through Saturday. "
            "On gazetted national holidays, the SLA clock for P3 and P4 tickets is paused "
            "at the end of the last business day before the holiday and automatically resumes "
            "at 9:00 AM on the next working day. "
            "Customers who submit non-urgent tickets outside business hours receive an "
            "automated acknowledgment message stating the expected next-response time "
            "so they are not left wondering when to expect a reply."
        ),
    },
    {
        "doc_id": "KB-007",
        "topic": "repeat_complaint_handling",
        "content": (
            "A support complaint is classified as a repeat complaint when the same customer "
            "submits tickets in the same issue category more than twice within any rolling "
            "30-day window. "
            "Repeat complaints bypass the normal Tier 1 queue and are automatically assigned "
            "to a Tier 2 agent within 2 hours of creation, regardless of their stated priority. "
            "The assigned Tier 2 agent is required to document a root-cause analysis in the "
            "ticket before it may be marked Resolved or Closed. "
            "Customers who accumulate three or more repeat complaints within a 90-day period "
            "are flagged for a proactive outreach call from an Ola customer-success manager "
            "to identify and address any underlying service deficiencies."
        ),
    },
    {
        "doc_id": "KB-008",
        "topic": "service_credit_policy",
        "content": (
            "Service credits are non-monetary, non-transferable compensations added directly "
            "to a customer's Ola wallet in response to confirmed service failures on Ola's "
            "part, such as SLA breaches, unjustified surge pricing, or verified driver misconduct. "
            "The standard credit amount ranges from INR 25 to INR 100, determined by a Tier 2 "
            "agent based on the severity and duration of the impact. "
            "Credits expire 90 days from the date of issuance and cannot be redeemed for cash "
            "or transferred to another account. "
            "Issuance of service credits above INR 100 requires approval from a Tier 3 manager "
            "and must be documented in the CRM with a justification note."
        ),
    },
    {
        "doc_id": "KB-009",
        "topic": "feedback_collection_process",
        "content": (
            "Immediately after each completed ride, Ola sends an automated in-app prompt "
            "asking the customer to rate their experience on a scale of 1 to 5 stars. "
            "Customers who submit a rating of 3 stars or below are shown a follow-up screen "
            "with an optional free-text field where they can describe the specific issue. "
            "All ride ratings and free-text comments are aggregated and reviewed daily by "
            "Ola's quality-assurance team to identify patterns in service degradation or "
            "driver behaviour. "
            "Drivers who maintain an average customer rating below 3.5 stars over their most "
            "recent 50 completed rides are automatically enrolled in Ola's mandatory driver "
            "performance-improvement programme."
        ),
    },
    {
        "doc_id": "KB-010",
        "topic": "vip_customer_handling",
        "content": (
            "Ola designates a customer as VIP when they meet all three criteria: a ride "
            "frequency of at least 20 rides per calendar month, an account tenure of at "
            "least 12 consecutive months, and a cumulative lifetime spend of at least "
            "INR 50,000. "
            "All support tickets submitted by VIP customers are automatically assigned P2 "
            "priority or above, irrespective of the stated issue category, and are routed "
            "to a dedicated VIP support queue. "
            "VIP agents must acknowledge the ticket within 30 minutes during business hours "
            "and within 2 hours outside business hours. "
            "Any confirmed service failure affecting a VIP customer results in an automatic "
            "service credit of INR 100, exceeding the standard credit amount, and may be "
            "supplemented by additional compensation at the discretion of the VIP account "
            "manager."
        ),
    },
    {
        "doc_id": "KB-011",
        "topic": "outage_communication_protocol",
        "content": (
            "When Ola's on-call engineering team confirms a P1 outage, the first public "
            "status update must be posted on the Ola Status Page within 15 minutes of "
            "incident confirmation, describing the affected services and the geographic "
            "regions impacted. "
            "Simultaneously, all riders and drivers in the affected area receive a push "
            "notification through the Ola app within 20 minutes of the incident declaration. "
            "The engineering team publishes a status refresh every 30 minutes on the Status "
            "Page and sends follow-up emails to enterprise account holders until the outage "
            "is fully resolved. "
            "Within 24 hours of resolution, Ola publishes a post-incident report detailing "
            "the root cause, the timeline of events, the customer impact, and the corrective "
            "actions taken to prevent recurrence."
        ),
    },
    {
        "doc_id": "KB-012",
        "topic": "data_retention_policy",
        "content": (
            "Ola retains all closed support tickets, including associated chat logs, "
            "attachments, and agent notes, for a mandatory period of 36 months from the "
            "date the ticket was marked Closed, in order to comply with applicable "
            "data-protection and consumer-rights regulations. "
            "During this retention window, ticket data is stored in an encrypted, "
            "role-access-controlled archive accessible only to authorised support staff "
            "and compliance auditors. "
            "After 36 months, all personally identifiable information within the ticket "
            "record is irreversibly purged, while anonymised statistical data (category, "
            "resolution time, escalation flag) is retained indefinitely for operational "
            "trend analysis. "
            "Customers may request early deletion of their personal data by submitting a "
            "Data Subject Access Request through the Ola privacy portal; such requests are "
            "fulfilled within 30 days unless retention is required by law."
        ),
    },
]

# Convenience dict for O(1) lookup by doc_id
KB_LOOKUP: dict[str, dict] = {d["doc_id"]: d for d in KB_DOCUMENTS}

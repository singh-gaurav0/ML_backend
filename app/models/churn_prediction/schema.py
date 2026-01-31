from pydantic import BaseModel

class ChurnRequest(BaseModel):
    total_mrr: float
    avg_seats: float
    upgrade_count: int
    downgrade_count: int
    auto_renew_ratio: float
    subscription_count: int
    days_since_last_subscription: int
    ticket_count: int
    avg_resolution_time: float
    avg_satisfaction: float
    escalation_ratio: float
    days_since_last_ticket: int
    active_subscription_at_cutoff: int
    tenure_days: int

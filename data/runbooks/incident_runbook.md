# Incident Response Runbook

## Severity levels
SEV1 is a complete outage or confirmed data breach and must be acknowledged within 5 minutes; an incident commander is assigned immediately. SEV2 is a major degradation affecting more than 20 percent of customers and must be acknowledged within 15 minutes. SEV3 is a partial degradation with a workaround and is acknowledged within 1 hour. SEV4 is cosmetic and handled in the backlog.

## Escalation
If a SEV1 is not mitigated within 60 minutes the incident commander pages the VP of Engineering. Regional outages are escalated to the Platform on-call team. Security events are escalated to the Security Operations team. Billing disputes are escalated to Finance Operations.

## Customer communication
For SEV1 and SEV2 incidents the communications lead posts a status page update every 30 minutes for SEV1 and every 60 minutes for SEV2. Customer messages must state what is affected, what we are doing, and when the next update is due. Never promise a fix time that engineering has not confirmed.

## Regional failover
Tier 1 services fail over to the secondary region. The operations lead confirms the regional failure, the incident commander approves failover, and traffic is drained from the failed region. Customers are notified within 10 minutes of the failover decision.

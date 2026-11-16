# API Integration Support Runbook

## HTTP 429 rate limit errors
A customer receiving HTTP 429 Too Many Requests has exceeded the per-key rate limit. Limits are 100 requests per minute on Basic, 1,000 on Business and 10,000 on Enterprise. First confirm the plan tier, then check whether the traffic was a one-off spike or sustained growth. Advise exponential backoff with jitter and honour the Retry-After header. Never advise immediate retries.

## Temporary limit increases
Business and Enterprise customers can get a temporary limit increase for a planned campaign. The request must be made at least 3 business days in advance and is approved by the support lead. Basic customers must upgrade their plan instead.

## HTTP 5xx errors
A burst of 502 or 503 responses on the tracking endpoints should be treated as a possible outage. Check the status page, then open a SEV2 if more than one region is affected or if more than 20 percent of requests fail. Attach request IDs from the customer logs.

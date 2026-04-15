# Test Results

Each run saves a JSON file here: `{benchmark}_{timestamp}.json`

Structure:
```json
{
  "benchmark": "reaction | mcgrawhill | site_stress",
  "timestamp": "2026-04-14T21:30:00",
  "backend": "claude | openai | ollama:model | local:model",
  "config_preset": "default | mcgrawhill",
  "metrics": { ... },
  "transition_log": [ ... ],
  "raw_observations": "optional, large"
}
```

Over time this directory builds a history of how DeltaVision performs
across backends, sites, and config tuning. No database needed — just
grep/jq the JSON files.

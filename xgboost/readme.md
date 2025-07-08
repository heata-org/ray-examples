# XGBoost on Ray-on-heata

1. Install ray
```
pip install ray
```

2. Signup for the Ray-on-heata beta https://www.heata.co/ray-sign-up

3. Use the credentials to connect to your cluster and submit the job

```
export RAY_ADDRESS='https://ray.heata.co/jobs'
export RAY_JOB_HEADERS='{"Authorization": "Bearer $YOUR_API_TOKEN"}'

ray job submit --runtime-env-json '{"pip": "./requirements.txt"}'  --working-dir . -- python xgboost-example.py
```

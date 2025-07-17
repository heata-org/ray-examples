# XGBoost demo on Heata with Ray

1. **[Click here to sign up for the Heata Ray beta](https://www.heata.co/ray-sign-up)**

2. **Set up Ray and use your credentials to connect to your cluster** as described on the [Heata Ray Dashboard](https://www.heata.co/ray-dashboard).

3. **Submit your job** with:
    ```
    ray job submit --runtime-env-json '{\"pip\": \"./requirements.txt\"}' --working-dir . -- python xgboost-example.py
    ```

---
**Any issues?** Please email [techsupport@heata.co](mailto:techsupport@heata.co)

..and, of course, 🙏 **thanks!**

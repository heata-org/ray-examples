# Parallel task demo on Heata with Ray

1. **[Click here to sign up for the Heata Ray beta](https://www.heata.co/ray-sign-up)**

2. **Set up Ray and use your credentials to connect to your cluster** as described on the [Heata Ray Dashboard](https://www.heata.co/ray-dashboard).

3. **Submit your job** with:

    ```
    ray job submit --runtime-env-json='{\"pip\": \"requirements.txt\"}' --no-wait --working-dir . -- python mandelbrot_zoom.py
    ```
4. **Wait** until done.. you can check with the commands output from the previous step. 

5. **Download the frames** with:
    ```
    pip install -r .\requirements.txt
    
    python -c "import mandelbrot_zoom as m; m.download_all_from_gcp('<JOB_SECRET_KEY>')"
    ```

    **Note:** Once the job complete it will output the above command with the correct <JOB_SECRET_KEY>

6. **Have a play** with the `CONFIG` section at the top of `mandelbrot_zoom.py` and submit again.


---
**Any issues?** Please email [techsupport@heata.co](mailto:techsupport@heata.co)

..and, of course, 🙏 **thanks!**

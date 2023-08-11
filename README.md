The repo contains 2 demos
1. `demo.py`
    - Compares PatchTST with NBEATS and NHITS
    - Compares PatchTST with ARIMA

2. `gradio_app/app.py`
    - A gradio app to interact with the forecasts
    - Run `python gradio_app/app.py` to start the app
    - The app can be accessed locally

The `data` folder has all the data used in the demos for convenience.

Some things to remember:
- Drop a `.env file` in the `gradio_app/` folder with your openai api key. This is required to use gpt4
- The first demo `demo.py` is best run on a GPU.
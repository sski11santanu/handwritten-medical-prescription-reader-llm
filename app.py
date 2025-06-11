import gradio as gr
from global_variables import VISION_MODELS, REASONING_MODEL_IDENTIFIER
from util_functions import analyze_prescription
import pandas as pd


def process(image, vision_model_name):
    # reasoning_model_identifier = VISION_MODELS[REASONING_MODEL_NAME]
    # vision_model_identifier = VISION_MODELS[vision_model_name]
    df = pd.read_csv("Generic + Medicine Names.csv")
    response = analyze_prescription(image, vision_model_name, REASONING_MODEL_IDENTIFIER, df)
    if not response:
        return "Error analyzing prescription. Please try again."
    response_string = str(response)
    print(response_string)
    return response_string

with gr.Blocks(theme = gr.themes.Soft(), title = "Handwritten Medical Prescription Reader") as demo:
    gr.Markdown("## Handwritten Medical Prescription Reader")

    with gr.Row():
        image_input = gr.Image(type = "pil", label = "Upload Image")
    
    with gr.Row():
        vision_dropdown = gr.Dropdown(choices = list(VISION_MODELS.keys()), label = "Choose Vision Model")
    
    submit_btn = gr.Button("Analyze")
    result_loading_indicator = gr.Markdown("### Result:\n", line_breaks = True)
    output_text = gr.Markdown(label = "Response", line_breaks = True)

    submit_btn.click(
        fn = process,
        inputs = [image_input, vision_dropdown],
        outputs = output_text
    )

# demo.launch(share = True, auth = ("prescription", "read_prescription_123"))
demo.launch()
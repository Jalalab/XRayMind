import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms
from groq import Groq
import base64
from io import BytesIO
import json

from model import MediScanModel, DISEASES, NUM_CLASSES
from gradcam import GradCAM

# ── Device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ── Load Model ──
print("Loading model...")
model = MediScanModel(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(
    torch.load('mediscan_densenet_best.pth', map_location=device, weights_only=False)
)
model.eval()
print("✅ Model loaded!")

# ── Grad-CAM ──
gradcam = GradCAM(model)

# ── Inference Transform ──
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

INFILTRATION_IDX = DISEASES.index('Infiltration')


# ── Medical Report Generator ──
class MedicalReportGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate(self, image, probs, diseases, threshold=0.2):
        high_findings = []
        low_findings = []

        for disease, prob in zip(diseases, probs):
            if prob >= 0.35:
                high_findings.append(f"- {disease}: {prob:.1%} (HIGH confidence)")
            elif prob >= threshold:
                low_findings.append(f"- {disease}: {prob:.1%} (LOW confidence)")

        if not high_findings and not low_findings:
            findings_text = "- No significant pathology detected"
        else:
            findings_text = ""
            if high_findings:
                findings_text += "High confidence findings:\n" + "\n".join(high_findings)
            if low_findings:
                findings_text += "\n\nLow confidence findings:\n" + "\n".join(low_findings)

        img_b64 = self._encode_image(image)

        prompt = f"""You are an expert radiologist assistant. Analyze this chest X-ray image along with the AI model predictions below.

AI Model Predictions:
{findings_text}

IMPORTANT INSTRUCTIONS:
- Always treat the TOP AI prediction as the primary finding
- Do NOT dismiss predictions as false positives
- The AI model is trained on 112,120 chest X-rays
- Even LOW confidence findings should be mentioned

Generate a structured radiology report. Use plain text only, NO asterisks, NO markdown:

CLINICAL INDICATION:
(Brief reason for the X-ray)

FINDINGS:
(Detailed observations referencing AI predictions)

IMPRESSION:
(Overall interpretation — name likely diagnosis)

SEVERITY:
(One of: Normal / Mild / Moderate / Severe)

RECOMMENDATIONS:
(Suggested next steps)

DISCLAIMER:
(Standard AI disclaimer)"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content


report_gen = MedicalReportGenerator()
print("✅ Report generator ready!")


def pil_to_base64(img):
    """Convert PIL image to base64 string for API response"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ── Full Analysis Pipeline ──
def analyze_xray(image_input):
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input).convert('RGB')
    else:
        pil_image = image_input.convert('RGB')

    tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()

    probs[INFILTRATION_IDX] *= 0.6

    predictions = {
        disease: float(prob)
        for disease, prob in zip(DISEASES, probs)
        if prob > 0.15
    }
    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

    top_idx = int(probs.argmax())
    cam = gradcam.generate(tensor, top_idx)
    heatmap = gradcam.overlay_heatmap(pil_image, cam)
    report = report_gen.generate(pil_image, probs, DISEASES)

    return {
        'predictions': predictions,
        'heatmap': heatmap,
        'report': report,
        'top_disease': DISEASES[top_idx],
        'top_prob': float(probs[top_idx]),
    }


# ── API Handler (used by github.io frontend) ──
# Returns JSON string with base64 heatmap
# so JavaScript can display results directly
def api_analyze(image):
    """
    API-friendly handler for github.io frontend.
    Returns:
      - heatmap_b64: base64 PNG of Grad-CAM overlay
      - predictions_json: JSON string of disease probabilities
      - report: plain text medical report
      - top_disease: string name of top finding
      - top_prob: float probability of top finding
      - status: success/error message
    """
    if image is None:
        return (
            "",
            "{}",
            "Please upload an image.",
            "No Finding",
            0.0,
            "error"
        )
    try:
        result = analyze_xray(image)

        # Convert heatmap PIL → base64 for JS
        heatmap_b64 = pil_to_base64(result['heatmap'])

        # Top 5 predictions as JSON string
        top5 = dict(list(result['predictions'].items())[:5])
        predictions_json = json.dumps(top5)

        status = f"✅ {result['top_disease']} ({result['top_prob']:.1%})"

        return (
            heatmap_b64,
            predictions_json,
            result['report'],
            result['top_disease'],
            result['top_prob'],
            status
        )
    except Exception as e:
        return ("", "{}", f"Error: {str(e)}", "Error", 0.0, f"❌ {str(e)}")


# ── Gradio UI Handler (original — kept for HuggingFace page) ──
def gradio_analyze(image):
    if image is None:
        return (None, "⚠️ No image uploaded", {}, "Please upload an image.", "⚠️ No image")
    try:
        result = analyze_xray(image)
        top5_preds = dict(list(result['predictions'].items())[:5])
        top_result = f"{result['top_disease']} ({result['top_prob']:.1%})"
        status = f"✅ Analysis complete! Top: {result['top_disease']} ({result['top_prob']:.1%})"
        return (np.array(result['heatmap']), top_result, top5_preds, result['report'], status)
    except Exception as e:
        err = f"❌ Error: {str(e)}"
        return (None, err, {}, err, err)


# ── Gradio App ──
with gr.Blocks(
    title="XRayMind — AI Radiology Assistant",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    css="""
    .header-box { text-align:center; padding:20px; border-bottom:1px solid #e2e8f0; margin-bottom:20px; }
    .disclaimer { font-size:0.85em; color:#94a3b8; text-align:center; }
    footer { display:none !important; }
    """
) as demo:

    gr.Markdown("""
    <div class="header-box">
    <h1 style="font-size:2.5em; font-weight:800; letter-spacing:-1px; margin:0;">
    XRayMind — AI Radiology Assistant
    </h1>
    <p style="color:#64748b; font-size:1.1em; margin-top:8px;">
    Chest X-Ray Analysis & Automated Report Generation
    </p>
    <p style="color:#94a3b8; font-size:0.85em; margin-top:4px;">
    Powered by DenseNet121 &nbsp;|&nbsp; Grad-CAM &nbsp;|&nbsp; LLaMA Medical Reporting
    </p>
    </div>
    """)

    gr.Markdown('<p class="disclaimer">⚠️ For research and educational purposes only. Not a substitute for professional medical diagnosis.</p>')
    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            image_input = gr.Image(label="Upload Chest X-Ray", type="pil", height=320)
            analyze_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=1):
            gr.Markdown("### 🔥 Grad-CAM Visualization")
            heatmap_output = gr.Image(label="Activation Heatmap", height=320)

    gr.Markdown("---")
    gr.Markdown("### 🎯 Primary Diagnosis")
    top_finding_output = gr.Textbox(label="Top Finding", interactive=False, lines=1)
    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Disease Probabilities")
            findings_output = gr.Label(label="Top 5 Findings", num_top_classes=5)

        with gr.Column(scale=2):
            gr.Markdown("### 📋 AI-Generated Radiology Report")
            report_output = gr.Textbox(label="Medical Report", lines=20, interactive=False)

    gr.Markdown("---")

    # ── Hidden API Interface ──
    # This is what github.io calls via JavaScript
    # It's invisible in the UI but accessible via API
    with gr.Row(visible=False):
        api_image = gr.Image(type="pil")
        api_heatmap = gr.Textbox()       # base64 PNG
        api_predictions = gr.Textbox()   # JSON string
        api_report = gr.Textbox()        # plain text
        api_top_disease = gr.Textbox()   # disease name
        api_top_prob = gr.Number()       # probability
        api_status = gr.Textbox()        # status

    gr.Markdown("""
    <p style="text-align:center; color:#94a3b8; font-size:0.85em; padding:10px;">
    Built by <strong>Jalal Abedin</strong> &nbsp;|&nbsp;
    DenseNet121 on NIH ChestX-ray14 (112,120 images) &nbsp;|&nbsp;
    North South University, Bangladesh
    </p>
    """)

    # Original UI button
    analyze_btn.click(
        fn=gradio_analyze,
        inputs=[image_input],
        outputs=[heatmap_output, top_finding_output, findings_output, report_output, status_box]
    )

    # API endpoint (called by github.io)
    api_image.change(
        fn=api_analyze,
        inputs=[api_image],
        outputs=[api_heatmap, api_predictions, api_report, api_top_disease, api_top_prob, api_status]
    )

demo.launch()

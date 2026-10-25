import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

def generate_pdf(output_filename: str = "Mask_RCNN_Commands_and_Shortkeys.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A2B4C"),
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2B6CB0"),
    )

    key_badge_style = ParagraphStyle(
        "KeyBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#7B341E"),
        alignment=1, # Center
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("Mask R-CNN GPU Webcam Detection", title_style))
    elements.append(Paragraph("Commands, Real-Time Shortcuts & Hyperparameters Guide", subtitle_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Hardware / Engine:</b> NVIDIA GeForce RTX 4060 GPU | PyTorch 2.6.0 (CUDA 12.4) | ResNet-50 FPN V2",
        ParagraphStyle("SubSub", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#718096"))
    ))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # 2. Keyboard Shortcuts Section
    elements.append(Paragraph("1. Real-Time Keyboard Shortcuts (Live Webcam HUD)", h1_style))
    elements.append(Paragraph("Press any of the following keys while the OpenCV webcam display window is focused:", body_style))
    elements.append(Spacer(1, 6))

    shortcut_data = [
        [
            Paragraph("<b>Key</b>", body_style),
            Paragraph("<b>Action Name</b>", body_style),
            Paragraph("<b>Description & Effect</b>", body_style),
            Paragraph("<b>Category</b>", body_style),
        ],
        [
            Paragraph("<b>M</b>", key_badge_style),
            Paragraph("Toggle Masks", body_style),
            Paragraph("Turn colored instance segmentation masks ON / OFF", body_style),
            Paragraph("Display", body_style),
        ],
        [
            Paragraph("<b>B</b>", key_badge_style),
            Paragraph("Toggle Boxes", body_style),
            Paragraph("Turn bounding box rectangle overlays ON / OFF", body_style),
            Paragraph("Display", body_style),
        ],
        [
            Paragraph("<b>L</b>", key_badge_style),
            Paragraph("Toggle Labels", body_style),
            Paragraph("Turn class names and confidence percentage pills ON / OFF", body_style),
            Paragraph("Display", body_style),
        ],
        [
            Paragraph("<b>+</b> or <b>=</b>", key_badge_style),
            Paragraph("Increase Conf", body_style),
            Paragraph("Raise confidence threshold by <b>+0.05</b> (up to 0.95)", body_style),
            Paragraph("Threshold", body_style),
        ],
        [
            Paragraph("<b>-</b> or <b>_</b>", key_badge_style),
            Paragraph("Decrease Conf", body_style),
            Paragraph("Lower confidence threshold by <b>-0.05</b> (down to 0.10)", body_style),
            Paragraph("Threshold", body_style),
        ],
        [
            Paragraph("<b>[</b>", key_badge_style),
            Paragraph("Decrease Opacity", body_style),
            Paragraph("Make segmentation masks more transparent (-0.05 alpha)", body_style),
            Paragraph("Styling", body_style),
        ],
        [
            Paragraph("<b>]</b>", key_badge_style),
            Paragraph("Increase Opacity", body_style),
            Paragraph("Make segmentation masks more opaque (+0.05 alpha)", body_style),
            Paragraph("Styling", body_style),
        ],
        [
            Paragraph("<b>S</b>", key_badge_style),
            Paragraph("Snapshot", body_style),
            Paragraph("Save high-res frame with detections to <code>captures/snapshot_*.png</code>", body_style),
            Paragraph("Capture", body_style),
        ],
        [
            Paragraph("<b>R</b>", key_badge_style),
            Paragraph("Record Video", body_style),
            Paragraph("Toggle MP4 video recording on/off (pulsing red REC indicator)", body_style),
            Paragraph("Capture", body_style),
        ],
        [
            Paragraph("<b>H</b>", key_badge_style),
            Paragraph("Help Modal", body_style),
            Paragraph("Open / close semi-transparent on-screen shortcuts help dialog", body_style),
            Paragraph("HUD", body_style),
        ],
        [
            Paragraph("<b>Q</b> / <b>ESC</b>", key_badge_style),
            Paragraph("Quit", body_style),
            Paragraph("Cleanly release camera, stop recording, and exit application", body_style),
            Paragraph("System", body_style),
        ],
    ]

    shortcut_table = Table(shortcut_data, colWidths=[65, 95, 290, 80])
    shortcut_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(shortcut_table)
    elements.append(Spacer(1, 14))

    # 3. Launch & CLI Commands Section
    elements.append(Paragraph("2. Launch & Execution Commands", h1_style))
    elements.append(Paragraph("You can launch the project using one-click scripts or terminal commands:", body_style))
    elements.append(Spacer(1, 6))

    cmd_data = [
        [
            Paragraph("<b>Scenario / Command</b>", body_style),
            Paragraph("<b>Command Syntax</b>", body_style),
            Paragraph("<b>Details</b>", body_style),
        ],
        [
            Paragraph("<b>1-Click Windows</b>", body_style),
            Paragraph("<code>run.bat</code>", code_style),
            Paragraph("Double-click in Explorer or run from CMD prompt", body_style),
        ],
        [
            Paragraph("<b>PowerShell</b>", body_style),
            Paragraph("<code>.\\run.ps1</code>", code_style),
            Paragraph("Direct execution in Windows PowerShell / pwsh", body_style),
        ],
        [
            Paragraph("<b>Default Launch</b>", body_style),
            Paragraph("<code>.venv\\Scripts\\python.exe main.py</code>", code_style),
            Paragraph("Loads all hyperparameters directly from <code>config.yaml</code>", body_style),
        ],
        [
            Paragraph("<b>Custom Config File</b>", body_style),
            Paragraph("<code>python main.py --config custom.yaml</code>", code_style),
            Paragraph("Use a specific YAML configuration preset", body_style),
        ],
        [
            Paragraph("<b>Change Resolution</b>", body_style),
            Paragraph("<code>python main.py --width 1920 --height 1080</code>", code_style),
            Paragraph("Overrides resolution to 1080p Full HD", body_style),
        ],
        [
            Paragraph("<b>Select Camera</b>", body_style),
            Paragraph("<code>python main.py --camera 1</code>", code_style),
            Paragraph("Connects to second / external USB camera index", body_style),
        ],
        [
            Paragraph("<b>Process Video File</b>", body_style),
            Paragraph("<code>python main.py --video sample.mp4</code>", code_style),
            Paragraph("Runs Mask R-CNN segmentation on a recorded video", body_style),
        ],
        [
            Paragraph("<b>Test GPU / CUDA</b>", body_style),
            Paragraph("<code>.venv\\Scripts\\python.exe test_gpu.py</code>", code_style),
            Paragraph("Runs CUDA tensor speed benchmark on RTX 4060", body_style),
        ],
    ]

    cmd_table = Table(cmd_data, colWidths=[120, 220, 190])
    cmd_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEFCBF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#744210")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFFAF0")]),
    ]))
    elements.append(cmd_table)
    elements.append(Spacer(1, 14))

    # 4. Hyperparameters Reference (config.yaml)
    elements.append(Paragraph("3. Central Hyperparameters Reference (config.yaml)", h1_style))
    elements.append(Paragraph("All runtime behavior is controlled by <b><code>config.yaml</code></b>. Edit this file to customize without touching any code:", body_style))
    elements.append(Spacer(1, 6))

    config_data = [
        [
            Paragraph("<b>Section</b>", body_style),
            Paragraph("<b>Hyperparameter</b>", body_style),
            Paragraph("<b>Default</b>", body_style),
            Paragraph("<b>Description & Valid Values</b>", body_style),
        ],
        [
            Paragraph("<b>model</b>", body_style),
            Paragraph("<code>version</code>", code_style),
            Paragraph("<code>\"v2\"</code>", body_style),
            Paragraph("Architecture version: <code>\"v2\"</code> (ResNet-50 FPN V2) or <code>\"v1\"</code>", body_style),
        ],
        [
            Paragraph("<b>model</b>", body_style),
            Paragraph("<code>device</code>", code_style),
            Paragraph("<code>\"cuda\"</code>", body_style),
            Paragraph("Hardware target: <code>\"cuda\"</code> (RTX GPU) or <code>\"cpu\"</code>", body_style),
        ],
        [
            Paragraph("<b>model</b>", body_style),
            Paragraph("<code>use_fp16</code>", code_style),
            Paragraph("<code>true</code>", body_style),
            Paragraph("Mixed precision inference for maximum FPS and lower VRAM", body_style),
        ],
        [
            Paragraph("<b>detection</b>", body_style),
            Paragraph("<code>conf_threshold</code>", code_style),
            Paragraph("<code>0.50</code>", body_style),
            Paragraph("Minimum confidence score to detect an object (0.05 to 0.95)", body_style),
        ],
        [
            Paragraph("<b>detection</b>", body_style),
            Paragraph("<code>mask_threshold</code>", code_style),
            Paragraph("<code>0.50</code>", body_style),
            Paragraph("Pixel probability cutoff for mask binarization (0.1 to 0.9)", body_style),
        ],
        [
            Paragraph("<b>detection</b>", body_style),
            Paragraph("<code>target_classes</code>", code_style),
            Paragraph("<code>[]</code>", body_style),
            Paragraph("List of class names to detect (e.g. <code>[\"person\", \"laptop\"]</code>). Empty = all", body_style),
        ],
        [
            Paragraph("<b>camera</b>", body_style),
            Paragraph("<code>index</code>", code_style),
            Paragraph("<code>0</code>", body_style),
            Paragraph("Webcam hardware index (0 = primary, 1 = USB webcam)", body_style),
        ],
        [
            Paragraph("<b>camera</b>", body_style),
            Paragraph("<code>width / height</code>", code_style),
            Paragraph("<code>1280 / 720</code>", body_style),
            Paragraph("Webcam capture resolution in pixels", body_style),
        ],
        [
            Paragraph("<b>visualization</b>", body_style),
            Paragraph("<code>mask_alpha</code>", code_style),
            Paragraph("<code>0.45</code>", body_style),
            Paragraph("Opacity of segmentation masks (0.10 transparent to 0.90 solid)", body_style),
        ],
        [
            Paragraph("<b>visualization</b>", body_style),
            Paragraph("<code>box_thickness</code>", code_style),
            Paragraph("<code>2</code>", body_style),
            Paragraph("Thickness in pixels for bounding box outlines", body_style),
        ],
        [
            Paragraph("<b>captures</b>", body_style),
            Paragraph("<code>save_dir</code>", code_style),
            Paragraph("<code>\"captures\"</code>", body_style),
            Paragraph("Folder path where snapshots and video recordings are stored", body_style),
        ],
    ]

    config_table = Table(config_data, colWidths=[80, 110, 75, 265])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(config_table)
    elements.append(Spacer(1, 12))

    # 5. COCO Supported Classes Reference
    elements.append(Paragraph("4. Supported COCO Classes (80 Categories)", h1_style))
    elements.append(Paragraph(
        "<b>People & Vehicles:</b> person, bicycle, car, motorcycle, airplane, bus, train, truck, boat<br/>"
        "<b>Outdoor & Signals:</b> traffic light, fire hydrant, stop sign, parking meter, bench<br/>"
        "<b>Animals:</b> bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe<br/>"
        "<b>Accessories & Travel:</b> backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket<br/>"
        "<b>Indoor & Electronics:</b> bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush",
        ParagraphStyle("ClassesStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=11.5, textColor=colors.HexColor("#4A5568"))
    ))

    # Build document
    doc.build(elements)
    print(f"[+] Successfully generated PDF: {output_filename}")
    return output_filename

if __name__ == "__main__":
    generate_pdf()

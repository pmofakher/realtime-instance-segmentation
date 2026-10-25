import os
import sys
import time
import datetime
import argparse
import cv2
import numpy as np

from config_loader import load_config
from mask_rcnn_detector import MaskRCNNDetector

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Mask R-CNN GPU Webcam Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to hyperparameters YAML configuration file")
    parser.add_argument("--camera", type=int, default=None, help="Override camera index from config")
    parser.add_argument("--video", type=str, default=None, help="Override video file path from config")
    parser.add_argument("--width", type=int, default=None, help="Override capture width from config")
    parser.add_argument("--height", type=int, default=None, help="Override capture height from config")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold from config")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if GPU is available")
    return parser.parse_args()


def open_camera(camera_idx: int, width: int, height: int, buffer_size: int = 1):
    """Opens webcam with optimal Windows DirectShow backend and buffer size."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
    else:
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def draw_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    object_count: int,
    conf_thresh: float,
    mask_alpha: float,
    draw_masks: bool,
    draw_boxes: bool,
    draw_labels: bool,
    device_name: str,
    is_recording: bool,
    show_help: bool,
    notification_msg: str = "",
):
    h, w = frame.shape[:2]

    # Top HUD Bar with semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 18, 24), -1)
    # Bottom shortcuts bar
    cv2.rectangle(overlay, (0, h - 34), (w, h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status pills on top bar
    gpu_badge = f"GPU: {device_name}"
    cv2.putText(frame, gpu_badge, (15, 32), font, 0.52, (100, 240, 160), 1, cv2.LINE_AA)

    perf_text = f"FPS: {fps:4.1f} | Latency: {inference_ms:4.1f}ms"
    cv2.putText(frame, perf_text, (w // 2 - 130, 32), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    stats_text = f"Detections: {object_count} | Conf: {conf_thresh:.2f}"
    cv2.putText(frame, stats_text, (w - 330, 32), font, 0.52, (230, 200, 90), 1, cv2.LINE_AA)

    # Recording badge
    if is_recording:
        blink = (int(time.time() * 2) % 2) == 0
        if blink:
            cv2.circle(frame, (w - 360, 28), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (w - 346, 33), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Bottom hotkeys bar
    status_masks = "ON" if draw_masks else "OFF"
    status_boxes = "ON" if draw_boxes else "OFF"
    status_labels = "ON" if draw_labels else "OFF"

    hotkey_str = (
        f"[Q] Quit  [M] Masks:{status_masks}  [B] Boxes:{status_boxes}  "
        f"[L] Labels:{status_labels}  [+/-] Conf  [[/]] Alpha:{mask_alpha:.2f}  "
        f"[S] Snapshot  [R] Record  [H] Help"
    )
    cv2.putText(frame, hotkey_str, (15, h - 11), font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    # Toast notification
    if notification_msg:
        toast_w = min(w - 60, len(notification_msg) * 12 + 40)
        tx1 = (w - toast_w) // 2
        tx2 = tx1 + toast_w
        cv2.rectangle(frame, (tx1, 65), (tx2, 105), (30, 30, 30), -1)
        cv2.rectangle(frame, (tx1, 65), (tx2, 105), (0, 200, 100), 2)
        cv2.putText(
            frame,
            notification_msg,
            (tx1 + 18, 92),
            font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Help modal
    if show_help:
        modal = frame.copy()
        mw, mh = 580, 370
        mx1, my1 = (w - mw) // 2, (h - mh) // 2
        mx2, my2 = mx1 + mw, my1 + mh

        cv2.rectangle(modal, (mx1, my1), (mx2, my2), (20, 20, 28), -1)
        cv2.rectangle(modal, (mx1, my1), (mx2, my2), (70, 130, 240), 2)
        cv2.addWeighted(modal, 0.92, frame, 0.08, 0, frame)

        title = "MASK R-CNN CONTROLS & SHORTKEYS"
        cv2.putText(frame, title, (mx1 + 25, my1 + 40), font, 0.65, (100, 200, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (mx1 + 25, my1 + 52), (mx2 - 25, my1 + 52), (70, 130, 240), 1)

        help_lines = [
            ("M", "Toggle Instance Segmentation Masks"),
            ("B", "Toggle Bounding Boxes"),
            ("L", "Toggle Class Labels & Confidence Scores"),
            ("+ or =", "Increase Confidence Threshold (+0.05)"),
            ("- or _", "Decrease Confidence Threshold (-0.05)"),
            ("[", "Decrease Mask Opacity (-0.05)"),
            ("]", "Increase Mask Opacity (+0.05)"),
            ("S", "Take Snapshot (saved to captures/)"),
            ("R", "Start / Stop MP4 Video Recording"),
            ("H", "Toggle this Help Window"),
            ("Q / ESC", "Quit Application"),
        ]

        y_offset = my1 + 82
        for key, desc in help_lines:
            cv2.putText(frame, f"[{key}]", (mx1 + 35, y_offset), font, 0.48, (255, 215, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (mx1 + 140, y_offset), font, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 24


def main():
    args = parse_args()

    # Load hyperparameters from config.yaml
    cfg = load_config(args.config)

    # Extract sections
    model_cfg = cfg.get("model", {})
    det_cfg = cfg.get("detection", {})
    cam_cfg = cfg.get("camera", {})
    vis_cfg = cfg.get("visualization", {})
    cap_cfg = cfg.get("captures", {})

    # CLI overrides if provided
    device = "cpu" if args.cpu else model_cfg.get("device", "cuda")
    conf_thresh = args.conf if args.conf is not None else float(det_cfg.get("conf_threshold", 0.50))
    cam_index = args.camera if args.camera is not None else int(cam_cfg.get("index", 0))
    video_source = args.video if args.video is not None else cam_cfg.get("video_file", None)
    cam_width = args.width if args.width is not None else int(cam_cfg.get("width", 1280))
    cam_height = args.height if args.height is not None else int(cam_cfg.get("height", 720))

    # Ensure captures output directory exists
    save_dir = cap_cfg.get("save_dir", "captures")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize detector using hyperparameters from config
    detector = MaskRCNNDetector(
        conf_threshold=conf_thresh,
        mask_threshold=float(det_cfg.get("mask_threshold", 0.50)),
        use_v2=(model_cfg.get("version", "v2").lower() == "v2"),
        use_fp16=bool(model_cfg.get("use_fp16", True)),
        device=device,
        warmup=bool(model_cfg.get("warmup", True)),
        target_classes=det_cfg.get("target_classes", []),
    )

    # Initialize video capture
    if video_source:
        print(f"[*] Opening video file: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[!] Error: Unable to open video file '{video_source}'")
            return
    else:
        print(f"[*] Connecting to webcam index {cam_index} ({cam_width}x{cam_height})...")
        cap = open_camera(cam_index, cam_width, cam_height, int(cam_cfg.get("buffer_size", 1)))
        if cap is None:
            print(f"[!] Error: Could not open camera {cam_index}.")
            print("    Check that your webcam is plugged in and not in use by another app.")
            return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Video feed active: {actual_w}x{actual_h}")
    print("[*] Press 'H' in the display window for hotkey help, or 'Q' to quit.")

    # Visualization state initialized from config
    draw_masks = bool(vis_cfg.get("draw_masks", True))
    draw_boxes = bool(vis_cfg.get("draw_boxes", True))
    draw_labels = bool(vis_cfg.get("draw_labels", True))
    show_hud = bool(vis_cfg.get("show_hud", True))
    mask_alpha = float(vis_cfg.get("mask_alpha", 0.45))
    box_thickness = int(vis_cfg.get("box_thickness", 2))
    contour_thickness = int(vis_cfg.get("contour_thickness", 2))
    font_scale = float(vis_cfg.get("font_scale", 0.52))

    show_help = False

    # Performance tracking
    fps_history = []
    fps_smooth = 0.0
    last_frame_time = time.perf_counter()

    # Recording state
    is_recording = False
    video_writer = None
    recording_fps = float(cap_cfg.get("recording_fps", 20.0))
    video_codec = str(cap_cfg.get("video_codec", "mp4v"))

    # Toast notifications
    notification_msg = ""
    notification_time = 0.0

    window_name = "Mask R-CNN GPU Webcam Detection - PyTorch"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, actual_w), min(720, actual_h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if video_source:
                    print("[*] End of video stream reached.")
                    break
                else:
                    print("[!] Failed to grab frame from webcam. Retrying...")
                    time.sleep(0.05)
                    continue

            # Update detector threshold
            detector.conf_threshold = conf_thresh

            # Run GPU detection
            results = detector.detect(frame)

            # Render detections
            annotated = detector.render(
                frame,
                results,
                draw_masks=draw_masks,
                draw_boxes=draw_boxes,
                draw_labels=draw_labels,
                mask_alpha=mask_alpha,
                box_thickness=box_thickness,
                contour_thickness=contour_thickness,
                font_scale=font_scale,
            )

            # Calculate smoothed FPS
            now = time.perf_counter()
            delta = now - last_frame_time
            last_frame_time = now
            if delta > 0:
                current_fps = 1.0 / delta
                fps_history.append(current_fps)
                if len(fps_history) > 15:
                    fps_history.pop(0)
                fps_smooth = sum(fps_history) / len(fps_history)

            # Check notification expiration (2.5 seconds)
            if notification_msg and (now - notification_time > 2.5):
                notification_msg = ""

            # Render HUD if enabled
            if show_hud:
                draw_hud(
                    annotated,
                    fps=fps_smooth,
                    inference_ms=results["inference_ms"],
                    object_count=len(results["boxes"]),
                    conf_thresh=conf_thresh,
                    mask_alpha=mask_alpha,
                    draw_masks=draw_masks,
                    draw_boxes=draw_boxes,
                    draw_labels=draw_labels,
                    device_name=detector.device_name,
                    is_recording=is_recording,
                    show_help=show_help,
                    notification_msg=notification_msg,
                )

            # Write to video recording if active
            if is_recording and video_writer is not None:
                video_writer.write(annotated)

            # Display window
            cv2.imshow(window_name, annotated)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key == ord("m"):
                draw_masks = not draw_masks
            elif key == ord("b"):
                draw_boxes = not draw_boxes
            elif key == ord("l"):
                draw_labels = not draw_labels
            elif key == ord("h"):
                show_help = not show_help
            elif key in (ord("+"), ord("=")):
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            elif key in (ord("-"), ord("_")):
                conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
            elif key == ord("["):
                mask_alpha = max(0.10, round(mask_alpha - 0.05, 2))
            elif key == ord("]"):
                mask_alpha = min(0.90, round(mask_alpha + 0.05, 2))
            elif key == ord("s"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(save_dir, f"snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, annotated)
                notification_msg = f"Snapshot saved: {os.path.basename(snap_path)}"
                notification_time = time.perf_counter()
                print(f"[*] Saved snapshot to {snap_path}")
            elif key == ord("r"):
                is_recording = not is_recording
                if is_recording:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(save_dir, f"recording_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*video_codec)
                    video_writer = cv2.VideoWriter(rec_path, fourcc, recording_fps, (actual_w, actual_h))
                    notification_msg = f"Recording started: {os.path.basename(rec_path)}"
                    notification_time = time.perf_counter()
                    print(f"[*] Started recording to {rec_path}")
                else:
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    notification_msg = "Recording saved!"
                    notification_time = time.perf_counter()
                    print("[*] Stopped recording.")

    finally:
        print("[*] Releasing resources...")
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Application terminated cleanly.")


if __name__ == "__main__":
    main()

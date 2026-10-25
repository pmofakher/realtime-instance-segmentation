import time
import colorsys
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

class MaskRCNNDetector:
    """
    GPU-accelerated Mask R-CNN detector with real-time segmentation and visualization.
    """

    def __init__(
        self,
        conf_threshold: float = 0.50,
        mask_threshold: float = 0.50,
        use_v2: bool = True,
        use_fp16: bool = True,
        device: str = "cuda",
        warmup: bool = True,
        target_classes: list = None,
    ):
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.use_fp16 = use_fp16

        # Determine target device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.use_fp16 = False  # FP16 autocast is intended for CUDA

        print(f"[*] Initializing Mask R-CNN on {self.device_name} (device: {self.device})")

        # Load weights and model
        if use_v2:
            print("[*] Loading Mask R-CNN ResNet-50 FPN V2 weights (COCO)...")
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn_v2(weights=weights)
        else:
            print("[*] Loading Mask R-CNN ResNet-50 FPN V1 weights (COCO)...")
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights)

        self.categories = weights.meta["categories"]
        self.model.to(self.device)
        self.model.eval()

        # Class name to ID map for filtering
        self.category_to_id = {cat.lower(): idx for idx, cat in enumerate(self.categories)}
        self.set_target_classes(target_classes)

        # Generate a distinct color palette for COCO classes
        self.colors = self._generate_palette(len(self.categories))

        # Warm-up pass to compile CUDA kernels and avoid first-frame lag
        if warmup:
            self._warmup()

    def set_target_classes(self, target_classes: list):
        """Converts class names or IDs to a set of category IDs to filter."""
        if not target_classes:
            self.target_class_ids = None
            return

        target_ids = set()
        for item in target_classes:
            if isinstance(item, int):
                target_ids.add(item)
            elif isinstance(item, str):
                item_lower = item.strip().lower()
                if item_lower in self.category_to_id:
                    target_ids.add(self.category_to_id[item_lower])
                else:
                    print(f"[!] Warning: Unknown COCO class name: '{item}'")
        self.target_class_ids = target_ids if target_ids else None
        if self.target_class_ids:
            names = [self.categories[i] for i in self.target_class_ids]
            print(f"[*] Filtering to {len(self.target_class_ids)} target classes: {', '.join(names)}")

    def _generate_palette(self, num_classes: int):
        """Generates bright, visually distinct BGR colors for each class."""
        colors = {}
        for idx in range(num_classes):
            hue = (idx * 0.618033988749895) % 1.0
            sat = 0.85 + (idx % 3) * 0.05
            val = 0.90 + (idx % 2) * 0.08
            r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 1.0), min(val, 1.0))
            colors[idx] = (int(b * 255), int(g * 255), int(r * 255))
        return colors

    def _warmup(self):
        """Runs a dummy frame through GPU to initialize CUDA and PyTorch kernels."""
        if self.device.type == "cuda":
            print("[*] Warming up GPU inference engine...")
            dummy = torch.zeros((3, 360, 640), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if self.use_fp16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        _ = self.model([dummy])
                else:
                    _ = self.model([dummy])
            torch.cuda.synchronize()
            print("[*] GPU warm-up complete.")

    def detect(self, frame_bgr: np.ndarray):
        """
        Runs instance segmentation inference on a single BGR frame.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            if self.device.type == "cuda" and self.use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = self.model([tensor])
                torch.cuda.synchronize()
            else:
                predictions = self.model([tensor])

        inference_ms = (time.perf_counter() - start_time) * 1000.0

        pred = predictions[0]
        scores = pred["scores"].detach()
        boxes = pred["boxes"].detach()
        labels = pred["labels"].detach()
        masks = pred["masks"].detach()

        # Filter by confidence threshold
        keep_mask = scores >= self.conf_threshold

        # Filter by target class IDs if configured
        if self.target_class_ids is not None:
            class_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            for c_id in self.target_class_ids:
                class_mask = class_mask | (labels == c_id)
            keep_mask = keep_mask & class_mask

        scores = scores[keep_mask].cpu().numpy()
        boxes = boxes[keep_mask].cpu().numpy().astype(int)
        labels = labels[keep_mask].cpu().numpy()

        # Binarize masks with mask_threshold
        if len(scores) > 0:
            masks = (masks[keep_mask] > self.mask_threshold).squeeze(1).cpu().numpy().astype(np.uint8)
        else:
            masks = np.empty((0, frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)

        class_names = [
            self.categories[idx] if idx < len(self.categories) else f"ID-{idx}"
            for idx in labels
        ]

        return {
            "boxes": boxes,
            "labels": labels,
            "class_names": class_names,
            "scores": scores,
            "masks": masks,
            "inference_ms": inference_ms,
        }

    def render(
        self,
        frame_bgr: np.ndarray,
        results: dict,
        draw_masks: bool = True,
        draw_boxes: bool = True,
        draw_labels: bool = True,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
        contour_thickness: int = 2,
        font_scale: float = 0.52,
    ) -> np.ndarray:
        """
        Renders segmentation masks, boundary contours, bounding boxes, and label pills.
        """
        output = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        boxes = results["boxes"]
        labels = results["labels"]
        scores = results["scores"]
        masks = results["masks"]
        class_names = results["class_names"]

        num_instances = len(boxes)
        if num_instances == 0:
            return output

        # Render colored masks and boundary contours
        if draw_masks and len(masks) > 0:
            mask_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
            contour_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(num_instances):
                mask = masks[i]
                color = self.colors.get(labels[i], (0, 255, 0))

                # Colorize mask
                mask_layer[mask == 1] = color

                # Compute smooth boundary contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_canvas, contours, -1, color, contour_thickness, cv2.LINE_AA)

            # Alpha blend mask with frame
            blended = cv2.addWeighted(output, 1.0, mask_layer, mask_alpha, 0)

            # Re-apply contours for crisp boundaries
            contour_mask = (contour_canvas > 0).any(axis=2)
            blended[contour_mask] = contour_canvas[contour_mask]
            output = blended

        # Render bounding boxes and modern label pills
        if draw_boxes or draw_labels:
            for i in range(num_instances):
                x1, y1, x2, y2 = boxes[i]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                color = self.colors.get(labels[i], (0, 255, 0))

                # Draw bounding box
                if draw_boxes:
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
                    # Sleek corner accents
                    line_len = min(20, max(5, (x2 - x1) // 5))
                    accent_th = max(box_thickness + 1, 3)
                    cv2.line(output, (x1, y1), (x1 + line_len, y1), (255, 255, 255), accent_th, cv2.LINE_AA)
                    cv2.line(output, (x1, y1), (x1, y1 + line_len), (255, 255, 255), accent_th, cv2.LINE_AA)
                    cv2.line(output, (x2, y2), (x2 - line_len, y2), (255, 255, 255), accent_th, cv2.LINE_AA)
                    cv2.line(output, (x2, y2), (x2 - line_len, y2), (255, 255, 255), accent_th, cv2.LINE_AA)

                # Draw label pill
                if draw_labels:
                    label_text = f"{class_names[i]} {int(scores[i] * 100)}%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

                    pill_y1 = max(0, y1 - text_h - 10)
                    pill_y2 = pill_y1 + text_h + 8
                    pill_x1 = x1
                    pill_x2 = min(w - 1, x1 + text_w + 14)

                    cv2.rectangle(output, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
                    # Shadow
                    cv2.putText(
                        output,
                        label_text,
                        (pill_x1 + 6, pill_y2 - 5),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + 1,
                        cv2.LINE_AA,
                    )
                    # Crisp text
                    cv2.putText(
                        output,
                        label_text,
                        (pill_x1 + 6, pill_y2 - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

        return output

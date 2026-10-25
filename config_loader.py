import os
import yaml

DEFAULT_CONFIG = {
    "model": {
        "version": "v2",
        "device": "cuda",
        "use_fp16": True,
        "warmup": True,
    },
    "detection": {
        "conf_threshold": 0.50,
        "mask_threshold": 0.50,
        "target_classes": [],
    },
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "video_file": None,
        "buffer_size": 1,
    },
    "visualization": {
        "draw_masks": True,
        "draw_boxes": True,
        "draw_labels": True,
        "show_hud": True,
        "mask_alpha": 0.45,
        "box_thickness": 2,
        "contour_thickness": 2,
        "font_scale": 0.52,
    },
    "captures": {
        "save_dir": "captures",
        "recording_fps": 20.0,
        "video_codec": "mp4v",
    },
}

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration from a YAML file, merging with defaults for any missing keys.
    """
    config = dict(DEFAULT_CONFIG)

    if not os.path.exists(config_path):
        print(f"[!] Warning: Config file '{config_path}' not found. Using built-in defaults.")
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user_config into defaults
        for section, sec_values in user_config.items():
            if section in config and isinstance(sec_values, dict):
                config[section].update(sec_values)
            else:
                config[section] = sec_values

        print(f"[*] Loaded hyperparameters from: {config_path}")
    except Exception as e:
        print(f"[!] Error reading '{config_path}': {e}. Falling back to default parameters.")

    return config

# target_specs.py
# 所有比例均相对于 image_long_side 或 image_area

TARGET_SPECS = {
    "cucumber": {
        "type": "vegetable",
        "hsv": {
            # hsv 绿色范围
            "lower": (35, 60, 40),
            "upper": (125, 255, 255),
            # hsv 绿色范围内的最小占比
            "min_ratio": 0.30
        },
        "geometry": {
            # 主轴长度范围 ≈ 图像长边的 1/4
            "min_length_ratio": 0.18,
            "max_length_ratio": 0.40,

            # 长宽比
            "min_aspect": 3.0,

            # 宽度不能太粗
            "max_width_ratio": 0.08
        }
    },

    "carrot": {
        "type": "vegetable",
        "hsv": {
            "lower": (5, 80, 80),
            "upper": (25, 255, 255),
            "min_ratio": 0.30
        },
        "geometry": {
            "min_length_ratio": 0.12,
            "min_aspect": 3.0
        }
    },

    "eggplant": {
        "type": "vegetable",
        "hsv": {
            "lower": (125, 30, 30),
            "upper": (160, 255, 255),
            "min_ratio": 0.30
        },
        "geometry": {
            "min_length_ratio": 0.12,
            "min_aspect": 3.0
        }
    },

    "box": {
        "type": "box",
        "hsv": {
            "lower": (100, 120, 80),
            "upper": (135, 255, 255),
            "min_ratio": 0.70
        },
        "geometry": {
            # 盒子一定是“大东西”
            "min_area_ratio": 0.15
        }
    }
}

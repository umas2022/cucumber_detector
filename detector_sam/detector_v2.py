'''
增加过滤规则：【绿色+长度小于阈值】，用于过滤萝卜叶
'''
import cv2
import numpy as np
from ultralytics import SAM


# =========================
# 颜色类别配置
# =========================
COLOR_CLASSES = {
    "green": {
        "hsv_lower": (30, 20, 20),
        "hsv_upper": (90, 255, 255),
        "min_ratio": 0.25
    },
    "red": {
        "hsv_lower": (0, 50, 50),
        "hsv_upper": (10, 255, 255),
        "min_ratio": 0.25
    },
    "yellow": {
        "hsv_lower": (15, 80, 80),
        "hsv_upper": (35, 255, 255),
        "min_ratio": 0.25
    },
        "purple": { 
        "hsv_lower": (150, 30, 20),  
        "hsv_upper": (255, 255, 255), 
        "min_ratio": 0.25 
    }
}


class ProduceDetector:
    """
    通用蔬果检测器
    SAM 分割 + 颜色语义 + PCA 主轴 + 宽度估计
    """

    def __init__(
        self,
        sam_model_path,
        color_classes,
        max_area_ratio=0.4,
        min_area_ratio=0.002,
        edge_margin=10,
        min_green_length=80,   # ← 新增：绿色最小长度，用于筛除萝卜叶
    ):
        self.model = SAM(sam_model_path)

        self.color_classes = {}
        for name, cfg in color_classes.items():
            self.color_classes[name] = {
                "lower": np.array(cfg["hsv_lower"], dtype=np.uint8),
                "upper": np.array(cfg["hsv_upper"], dtype=np.uint8),
                "min_ratio": cfg.get("min_ratio", 0.2)
            }

        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.edge_margin = edge_margin
        self.min_green_length = min_green_length

    # =========================
    # 主接口
    # =========================
    def detect(self, image_input, visualize=False):
        img = self._load_image(image_input)
        H, W = img.shape[:2]

        results = self.model(img)
        result = results[0]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        vis_img = img.copy().astype(np.float32)

        objects = []

        if result.masks is None or result.boxes is None:
            return {"success": False, "objects": []}

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for mask, box in zip(masks, boxes):
            if not self._pass_geometry_filter(mask, box, H, W):
                continue

            mask_bool = mask.astype(bool)

            # ---- 颜色分类 ----
            class_name, class_ratio = self._classify_color(mask_bool, hsv)
            if class_name is None:
                continue

            # ---- 主轴、端点、宽度 ----
            head, tail, axis, center, width = self._axis_endpoints_and_width(mask_bool)

            # ---- 计算长度 ----
            length = float(np.linalg.norm(head - tail))

            # =========================
            # 叶子过滤规则（核心新增）
            # =========================
            if class_name == "green" and length < self.min_green_length:
                # 判定为绿色叶子，直接跳过
                continue

            obj = {
                "class": class_name,
                "color_ratio": class_ratio,
                "head": tuple(head),
                "tail": tuple(tail),
                "center": tuple(center),
                "axis": tuple(axis),
                "length": length,
                "width": width,
                "mask": mask_bool
            }
            objects.append(obj)

            if visualize:
                self._draw(
                    vis_img,
                    mask_bool,
                    head,
                    tail,
                    class_name,
                    axis,
                    center,
                    width
                )

        output = {
            "success": len(objects) > 0,
            "objects": objects
        }

        if visualize:
            output["vis_image"] = vis_img.astype(np.uint8)

        return output

    # =========================
    # 内部方法
    # =========================
    def _load_image(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"无法读取图像: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError("image_input 必须是 str 或 np.ndarray")

    def _pass_geometry_filter(self, mask, box, H, W):
        area_ratio = mask.sum() / (H * W)
        if area_ratio > self.max_area_ratio or area_ratio < self.min_area_ratio:
            return False

        x1, y1, x2, y2 = box
        if (
            x1 < self.edge_margin or y1 < self.edge_margin or
            (W - x2) < self.edge_margin or (H - y2) < self.edge_margin
        ):
            return False

        return True

    def _classify_color(self, mask_bool, hsv):
        best_class = None
        best_ratio = 0.0

        for name, cfg in self.color_classes.items():
            color_mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
            ratio = (color_mask[mask_bool] > 0).sum() / mask_bool.sum()

            if ratio >= cfg["min_ratio"] and ratio > best_ratio:
                best_class = name
                best_ratio = ratio

        return best_class, best_ratio

    def _axis_endpoints_and_width(self, mask_bool):
        ys, xs = np.where(mask_bool)
        points = np.stack([xs, ys], axis=1).astype(np.float32)

        center = points.mean(axis=0)
        centered = points - center

        # ---- PCA 主轴（SVD）----
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        axis /= np.linalg.norm(axis)

        # ---- 端点 ----
        projections = centered @ axis
        head = points[np.argmax(projections)].astype(int)
        tail = points[np.argmin(projections)].astype(int)

        # ---- 法向宽度（夹爪张角依据）----
        normal = np.array([-axis[1], axis[0]])
        normal /= np.linalg.norm(normal)

        proj_n = centered @ normal
        width = float(proj_n.max() - proj_n.min())

        return head, tail, axis, center.astype(int), width

    def _draw(self, vis_img, mask_bool, head, tail, class_name, axis, center, width):
        color_map = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255)
        }
        color = np.array(color_map.get(class_name, (255, 255, 255)), dtype=np.uint8)

        # mask
        vis_img[mask_bool] = vis_img[mask_bool] * 0.5 + color * 0.5

        # 端点
        cv2.circle(vis_img, tuple(head), 8, (255, 255, 255), -1)
        cv2.circle(vis_img, tuple(tail), 8, (0, 0, 0), -1)
        cv2.line(vis_img, tuple(head), tuple(tail), (255, 255, 255), 2)

        # 宽度方向（夹爪示意）
        normal = np.array([-axis[1], axis[0]])
        p1 = (center + normal * width / 2).astype(int)
        p2 = (center - normal * width / 2).astype(int)
        cv2.line(vis_img, tuple(p1), tuple(p2), (255, 0, 255), 2)


# =========================
# 示例运行
# =========================
if __name__ == "__main__":

    detector = ProduceDetector(
        sam_model_path="sam2.1_b.pt",
        color_classes=COLOR_CLASSES,
    )

    # result = detector.detect("./img/2025-12-19-095730.jpg", visualize=True)
    result = detector.detect("./img/2025-12-19-095745.jpg", visualize=True)

    if result["success"]:
        for i, obj in enumerate(result["objects"]):
            print(
                f"[{i}] class={obj['class']}, "
                f"head={obj['head']}, "
                f"tail={obj['tail']}, "
                f"center={obj['center']}, "
                f"width(px)={obj['width']:.2f}, "
                f"axis={obj['axis']}"
            )

        cv2.imshow("Result", result["vis_image"])
        cv2.imwrite("./result/result.jpg", result["vis_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

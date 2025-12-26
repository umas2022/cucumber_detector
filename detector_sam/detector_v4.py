'''
蓝色筐子意外的区域添加黑色遮罩
python>=3.10
pip install opencv-python numpy ultralytics
'''
import cv2
import numpy as np
import time
from ultralytics import SAM
from target_specs import TARGET_SPECS


import cv2
import numpy as np
import time
from ultralytics import SAM
from target_specs import TARGET_SPECS


class ProduceSystemV5_BoxAware:

    def __init__(
        self,
        sam_model_path,
        min_area_ratio=0.002,
        max_area_ratio=0.6,
        edge_margin_ratio=0.02
    ):
        self.sam = SAM(sam_model_path)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.edge_margin_ratio = edge_margin_ratio

    # ==================================================
    # 对外主接口（推荐用这个）
    def run(self, image, visualize=False):
        return self.detect_in_box(image, visualize)

    # ==================================================
    # 第一阶段 + 第二阶段整合
    def detect_in_box(self, image, visualize=False):

        # ---------- 第 1 次检测：找 box ----------
        first = self.detect_all(image, visualize=False)

        boxes = [o for o in first["objects"] if o["type"] == "box"]
        if not boxes:
            return {
                "success": False,
                "objects": [],
                "reason": "no box detected"
            }

        box = boxes[0]  # 默认只有一个框

        # ---------- 抹掉框外 ----------
        boxed_img, box_mask = self._mask_outside_box(image, box["corners"])

        # ---------- 第 2 次检测：只找蔬菜 ----------
        second = self.detect_all(boxed_img, visualize=visualize)

        # 过滤掉 box 本身
        produce = [o for o in second["objects"] if o["type"] != "box"]

        return {
            "success": len(produce) > 0,
            "box": box,
            "objects": produce,
            "vis_image": second.get("vis_image")
        }

    # ==================================================
    # 原始完整检测逻辑（几乎等同你 V4_2 的 run）
    def detect_all(self, image, visualize=False):
        H, W = image.shape[:2]
        long_side = max(H, W)
        img_area = H * W

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        vis = image.copy().astype(np.float32)

        t0 = time.time()
        results = self.sam(image)
        sam_time = time.time() - t0

        if results[0].masks is None:
            return {"objects": []}

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        objects = []

        for mask, box in zip(masks, boxes):
            mask = mask.astype(bool)

            if not self._basic_filter(mask, box, H, W, img_area):
                continue

            geom = self._compute_geometry(mask, long_side, img_area)

            target = self._classify(mask, geom, hsv)
            if target is None:
                continue

            obj = {
                **target,
                **geom,
                "mask": mask
            }

            if obj["type"] == "box":
                obj.update(self._compute_box_corners(mask))

            objects.append(obj)

            if visualize:
                self._draw(vis, obj)

        return {
            "objects": objects,
            "sam_time": sam_time,
            "vis_image": vis.astype(np.uint8) if visualize else None
        }

    # ==================================================
    # 抹掉框外
    def _mask_outside_box(self, image, corners):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        result = image.copy()
        result[mask == 0] = (0, 0, 0)   # 或 (128,128,128)

        return result, mask

    # ==================================================
    def _basic_filter(self, mask, box, H, W, img_area):
        area_ratio = mask.sum() / img_area
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        margin = int(self.edge_margin_ratio * max(H, W))
        x1, y1, x2, y2 = box
        if x1 < margin or y1 < margin or (W - x2) < margin or (H - y2) < margin:
            return False

        return True

    # ==================================================
    def _compute_geometry(self, mask, long_side, img_area):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        center = pts.mean(axis=0)
        centered = pts - center

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]

        proj = centered @ axis
        head = pts[np.argmin(proj)]
        tail = pts[np.argmax(proj)]

        length = proj.max() - proj.min()
        normal = np.array([-axis[1], axis[0]])
        width = np.percentile(centered @ normal, 90) - np.percentile(centered @ normal, 10)

        return {
            "center": tuple(center.astype(int)),
            "head": tuple(head.astype(int)),
            "tail": tuple(tail.astype(int)),
            "axis": tuple(axis.tolist()),
            "length_ratio": float(length / long_side),
            "width_ratio": float(width / long_side),
            "aspect": float(length / (width + 1e-6)),
            "area_ratio": float(mask.sum() / img_area)
        }

    # ==================================================
    def _compute_box_corners(self, mask):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)

        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
        else:
            rect = cv2.minAreaRect(pts.astype(np.float32))
            corners = cv2.boxPoints(rect).astype(int)

        return {
            "corners": [tuple(p) for p in corners],
            "box_center": tuple(np.mean(corners, axis=0).astype(int))
        }

    # ==================================================
    def _classify(self, mask, geom, hsv):
        candidates = []

        for name, spec in TARGET_SPECS.items():
            if not self._geometry_pass(geom, spec["geometry"]):
                continue

            ratio = self._color_ratio(mask, hsv, spec["hsv"])
            if ratio >= spec["hsv"]["min_ratio"]:
                candidates.append((name, spec["type"], ratio))

        if not candidates:
            return None

        name, t, r = max(candidates, key=lambda x: x[2])
        return {"type": t, "subtype": name, "color_ratio": r}

    # ==================================================
    def _geometry_pass(self, geom, rule):
        for k, v in rule.items():
            if k == "min_length_ratio" and geom["length_ratio"] < v:
                return False
            if k == "max_length_ratio" and geom["length_ratio"] > v:
                return False
            if k == "min_aspect" and geom["aspect"] < v:
                return False
            if k == "max_width_ratio" and geom["width_ratio"] > v:
                return False
            if k == "min_area_ratio" and geom["area_ratio"] < v:
                return False
        return True

    # ==================================================
    def _color_ratio(self, mask, hsv, cfg):
        lower = np.array(cfg["lower"], np.uint8)
        upper = np.array(cfg["upper"], np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
        return (color_mask[mask] > 0).sum() / mask.sum()

    # ==================================================
    def _draw(self, img, obj):
        colors = {
            "cucumber": (0, 255, 0),
            "carrot": (0, 165, 255),
            "box": (255, 0, 0)
        }

        c = np.array(colors.get(obj["subtype"], (255, 255, 255)), np.uint8)
        img[obj["mask"]] = img[obj["mask"]] * 0.5 + c * 0.5

        if obj["type"] == "box":
            for p in obj["corners"]:
                cv2.circle(img, p, 6, (255, 255, 255), -1)
            for i in range(4):
                cv2.line(img, obj["corners"][i],
                         obj["corners"][(i + 1) % 4],
                         (255, 255, 255), 2)
            return

        cv2.circle(img, obj["center"], 5, (255, 255, 255), -1)
        cv2.circle(img, obj["head"], 6, (0, 0, 255), -1)
        cv2.circle(img, obj["tail"], 6, (255, 0, 0), -1)
        cv2.line(img, obj["head"], obj["tail"], (255, 255, 255), 2)


# ==================================================
if __name__ == "__main__":
    # img = cv2.imread("./img/2025-12-25-105108.jpg")
    # img = cv2.imread("./img/2025-12-25-105142.jpg")
    # img = cv2.imread("./img/2025-12-25-105214.jpg")
    img = cv2.imread("./img/2025-12-25-105257.jpg")
    # img = cv2.imread("./img/2025-12-25-105257_m.jpg")

    if img is None:
        raise RuntimeError("Failed to load image")

    system = ProduceSystemV5_BoxAware(
        sam_model_path="sam2.1_b.pt"
    )

    result = system.run(img, visualize=True)

    # ---------- 蓝色筐子 ----------
    box = result["box"]
    corners = box["corners"]
    box_center = box["box_center"]

    print("\n=== BOX ===")
    print("center:", box_center)
    print("corners:")
    for i, p in enumerate(corners):
        print(f"  {i}: {p}")

    # ---------- 蔬菜 ----------
    print("\n=== PRODUCE ===")
    for i, obj in enumerate(result["objects"]):
        print(
            f"[{i}] type={obj['type']} "
            f"subtype={obj['subtype']} "
            f"center={obj['center']} "
            f"head={obj['head']} "
            f"tail={obj['tail']} "
            f"aspect={obj['aspect']:.2f}"
        )

    if result["vis_image"] is not None:
        cv2.imshow("Produce Detection Result", result["vis_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np
import time
from ultralytics import SAM
from target_specs import TARGET_SPECS


class ProduceSystemV4_2:

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

    # --------------------------------------------------
    def run(self, image, visualize=False):
        H, W = image.shape[:2]
        long_side = max(H, W)
        img_area = H * W

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        vis = image.copy().astype(np.float32)

        t0 = time.time()
        results = self.sam(image)
        sam_time = time.time() - t0

        if results[0].masks is None:
            return {"success": False, "objects": []}

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

            # box 特殊处理：四角
            if obj["type"] == "box":
                obj.update(self._compute_box_corners(mask))

            objects.append(obj)

            if visualize:
                self._draw(vis, obj)

        return {
            "success": len(objects) > 0,
            "objects": objects,
            "sam_time": sam_time,
            "vis_image": vis.astype(np.uint8) if visualize else None
        }

    # --------------------------------------------------
    def _basic_filter(self, mask, box, H, W, img_area):
        area_ratio = mask.sum() / img_area
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        margin = int(self.edge_margin_ratio * max(H, W))
        x1, y1, x2, y2 = box
        if x1 < margin or y1 < margin or (W - x2) < margin or (H - y2) < margin:
            return False

        return True

    # --------------------------------------------------
    # 主轴 / 头尾（用于蔬菜）
    def _compute_geometry(self, mask, long_side, img_area):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        center = pts.mean(axis=0)
        centered = pts - center

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]

        proj = centered @ axis
        i_min = np.argmin(proj)
        i_max = np.argmax(proj)

        head = pts[i_min]
        tail = pts[i_max]

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

    # --------------------------------------------------
    # box 四角
    def _compute_box_corners(self, mask):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)

        # 1. 凸包（真实外轮廓）
        hull = cv2.convexHull(pts)

        # 2. 多边形近似
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        # 3. 理想情况：4 个点（梯形 / 四边形）
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)

        else:
            # fallback：最小外接矩形（极端失败情况）
            rect = cv2.minAreaRect(pts.astype(np.float32))
            corners = cv2.boxPoints(rect).astype(int)

        center = tuple(np.mean(corners, axis=0).astype(int))

        return {
            "box_center": center,
            "corners": [tuple(p) for p in corners]
        }


    # --------------------------------------------------
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

    # --------------------------------------------------
    def _geometry_pass(self, geom, rule):
        if "min_length_ratio" in rule and geom["length_ratio"] < rule["min_length_ratio"]:
            return False
        if "max_length_ratio" in rule and geom["length_ratio"] > rule["max_length_ratio"]:
            return False
        if "min_aspect" in rule and geom["aspect"] < rule["min_aspect"]:
            return False
        if "max_width_ratio" in rule and geom["width_ratio"] > rule["max_width_ratio"]:
            return False
        if "min_area_ratio" in rule and geom["area_ratio"] < rule["min_area_ratio"]:
            return False
        return True

    # --------------------------------------------------
    def _color_ratio(self, mask, hsv, cfg):
        lower = np.array(cfg["lower"], np.uint8)
        upper = np.array(cfg["upper"], np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
        return (color_mask[mask] > 0).sum() / mask.sum()

    # --------------------------------------------------
    def _draw(self, img, obj):
        colors = {
            "cucumber": (0, 255, 0),
            "carrot": (0, 165, 255),
            "box": (255, 0, 0)
        }

        c = np.array(colors.get(obj["subtype"], (255, 255, 255)), np.uint8)
        img[obj["mask"]] = img[obj["mask"]] * 0.5 + c * 0.5

        # ---- box：画四角 ----
        if obj["type"] == "box":
            for p in obj["corners"]:
                cv2.circle(img, p, 6, (255, 255, 255), -1)

            for i in range(4):
                cv2.line(
                    img,
                    obj["corners"][i],
                    obj["corners"][(i + 1) % 4],
                    (255, 255, 255),
                    2
                )
            return

        # ---- 蔬菜：头尾 + 轴 ----
        cv2.circle(img, obj["center"], 5, (255, 255, 255), -1)
        cv2.circle(img, obj["head"], 6, (0, 0, 255), -1)
        cv2.circle(img, obj["tail"], 6, (255, 0, 0), -1)
        cv2.line(img, obj["head"], obj["tail"], (255, 255, 255), 2)


# ==================================================
if __name__ == "__main__":
    img = cv2.imread("./img/2025-12-25-105142.jpg")
    if img is None:
        raise RuntimeError("Failed to load image")

    system = ProduceSystemV4_2(
        sam_model_path="sam2.1_b.pt"
    )

    result = system.run(img, visualize=True)

    print(f"SAM inference time: {result['sam_time']:.3f}s")

    for i, obj in enumerate(result["objects"]):
        print(
            f"[{i}] type={obj['type']} "
            f"subtype={obj['subtype']} "
            f"center={obj.get('center')} "
            f"head={obj.get('head')} "
            f"tail={obj.get('tail')} "
            f"corners={obj.get('corners')}"
        )

    if result["vis_image"] is not None:
        cv2.imshow("Produce Detection Result", result["vis_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

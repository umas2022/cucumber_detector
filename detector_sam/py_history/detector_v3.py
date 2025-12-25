import cv2
import numpy as np
import time
from ultralytics import SAM


# =========================
# 颜色规则（HSV）
# =========================

DEBUG = True
DEBUG_DRAW_REJECTED = True  # 是否把被拒绝的 mask 也画出来


COLOR_RULES = {
    "green": {   # 黄瓜 / 萝卜叶
        "lower": (30, 30, 30),
        "upper": (90, 255, 255),
        "min_ratio": 0.25
    },
    "orange": {  # 胡萝卜
        "lower": (5, 80, 80),
        "upper": (25, 255, 255),
        "min_ratio": 0.25
    },
    "purple": {  # 茄子
        "lower": (125, 30, 30),
        "upper": (160, 255, 255),
        "min_ratio": 0.25
    },
    "blue": {    # 蓝色盒子
        "lower": (90, 50, 50),
        "upper": (130, 255, 255),
        "min_ratio": 0.35
    }
}


# =========================
# 主系统
# =========================

class SamProduceSystem:

    def __init__(
        self,
        sam_model_path,
        max_area_ratio=0.5,
        min_area_ratio=0.002,
        edge_margin=20
    ):
        self.sam = SAM(sam_model_path)

        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.edge_margin = edge_margin

        self.color_rules = {
            k: {
                "lower": np.array(v["lower"], np.uint8),
                "upper": np.array(v["upper"], np.uint8),
                "min_ratio": v["min_ratio"]
            }
            for k, v in COLOR_RULES.items()
        }

    # =========================
    # 主入口
    # =========================
    def run(self, image, visualize=False):
        H, W = image.shape[:2]

        # ---- 1. SAM ----
        t0 = time.time()
        results = self.sam(image)
        sam_time = time.time() - t0

        result = results[0]
        if result.masks is None:
            return {"success": False, "objects": []}

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        vis = image.copy().astype(np.float32)

        objects = []

        # ---- 2. 遍历 mask ----
        # for mask, box in zip(masks, boxes):
        for idx, (mask, box) in enumerate(zip(masks, boxes)):

            mask_bool = mask.astype(bool)

            if not self._reject_background(mask_bool, box, H, W):
                if DEBUG:
                    print(f"[mask {idx}] rejected at BACKGROUND "
                        f"area_ratio={mask_bool.sum()/(H*W):.4f}")
                continue

            # ---- 3. 颜色分类 ----
            target, ratios = self._classify_target(mask_bool, hsv)
            if target is None:
                if DEBUG:
                    print(f"[mask {idx}] rejected at COLOR ratios={ratios}")
                if DEBUG_DRAW_REJECTED:
                    vis[mask_bool] = (0, 0, 255)  # 红色：颜色不过
                continue

            # ---- 4. 几何计算 ----
            if target["type"] == "box":
                geom = self._compute_box_geometry(mask_bool)

                if not self._filter_box_by_shape(mask_bool, geom):
                    if DEBUG:
                        print(f"[mask {idx}] blue rejected by BOX_SHAPE "
                            f"ratios={ratios}")
                    if DEBUG_DRAW_REJECTED:
                        vis[mask_bool] = (255, 255, 0)  # 黄色
                    continue

            else:
                geom = self._compute_axis(mask_bool)

                # ---- 4.1 绿色细长度过滤（去萝卜叶）----
                if target["subtype"] == "green":
                    elong = geom["length"] / max(geom["width"], 1e-6)

                    if DEBUG:
                        print(f"[mask {idx}] green elongation={elong:.2f} "
                            f"length={geom['length']:.1f} width={geom['width']:.1f}")

                    if elong < 4.0:
                        if DEBUG:
                            print(f"[mask {idx}] rejected at GREEN_SHAPE")
                        if DEBUG_DRAW_REJECTED:
                            vis[mask_bool] = (255, 0, 255)  # 紫色
                        continue

            obj = {
                "type": target["type"],
                "subtype": target["subtype"],
                "color_ratio": target["ratio"],
                **geom,
                "mask": mask_bool
            }

            objects.append(obj)

            if visualize:
                self._draw(vis, obj)

        out = {
            "success": len(objects) > 0,
            "objects": objects,
            "sam_time": sam_time
        }

        if visualize:
            out["vis_image"] = vis.astype(np.uint8)

        return out

    # =========================
    # 背景剔除
    # =========================
    def _reject_background(self, mask, box, H, W):
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

    # =========================
    # 颜色分类
    # =========================
    def _classify_target(self, mask, hsv):
        best = None
        ratios = {}

        for name, cfg in self.color_rules.items():
            color_mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
            ratio = (color_mask[mask] > 0).sum() / mask.sum()
            ratios[name] = ratio

            if ratio >= cfg["min_ratio"]:
                if best is None or ratio > best["ratio"]:
                    best = {
                        "type": "box" if name == "blue" else "vegetable",
                        "subtype": name,
                        "ratio": ratio
                    }

        return best, ratios

    # =========================
    # 蔬菜几何（PCA）
    # =========================
    def _compute_axis(self, mask):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        center = pts.mean(axis=0)
        centered = pts - center

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        axis /= np.linalg.norm(axis)

        proj = centered @ axis
        head = pts[np.argmax(proj)].astype(int)
        tail = pts[np.argmin(proj)].astype(int)

        normal = np.array([-axis[1], axis[0]])
        proj_n = centered @ normal

        # 使用鲁棒宽度，去掉极端点
        low = np.percentile(proj_n, 10)
        high = np.percentile(proj_n, 90)
        width = float(high - low)


        return {
            "center": tuple(center.astype(int)),
            "head": tuple(head),
            "tail": tuple(tail),
            "axis": tuple(axis),
            "width": width,
            "length": float(proj.max() - proj.min())
        }

    # =========================
    # 绿色目标形状过滤（黄瓜）
    # =========================
    def _filter_green_by_shape(self, geom):
        length = geom["length"]
        width = geom["width"]

        if width < 1e-3:
            return False

        elongation = length / width

        # ---- 可调参数 ----
        return elongation >= 3.0
    
    # =========================
    # 蓝色盒子形状过滤
    # =========================
    def _filter_box_by_shape(self, mask, geom):
        """
        蓝色目标二次验证，防止把黄瓜当盒子
        """
        area = mask.sum()
        box_area = cv2.contourArea(
            np.array(geom["box_points"], np.int32)
        )

        # mask 填充率（盒子应接近实心）
        fill_ratio = area / (box_area + 1e-6)

        # 长宽比（盒子不应极端细长）
        w = np.linalg.norm(
            np.array(geom["box_points"][0]) -
            np.array(geom["box_points"][1])
        )
        h = np.linalg.norm(
            np.array(geom["box_points"][1]) -
            np.array(geom["box_points"][2])
        )
        aspect = max(w, h) / (min(w, h) + 1e-6)

        if fill_ratio < 0.6:
            return False

        if aspect > 2.5:
            return False

        return True


    # =========================
    # 蓝色盒子几何
    # =========================
    def _compute_box_geometry(self, mask):
        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)

        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)

        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        # 若不是 4 点，退化为最小外接矩形
        if len(approx) != 4:
            rect = cv2.minAreaRect(pts.astype(np.float32))
            box = cv2.boxPoints(rect).astype(int)
            corners = box
        else:
            corners = approx.reshape(-1, 2)

        center = tuple(np.mean(corners, axis=0).astype(int))

        return {
            "center": center,
            "box_points": [tuple(p) for p in corners]
        }


    # =========================
    # 可视化
    # =========================
    def _draw(self, img, obj):
        color_map = {
            "green": (0, 255, 0),
            "orange": (0, 165, 255),
            "purple": (200, 0, 200),
            "blue": (255, 0, 0)
        }

        c = np.array(color_map.get(obj["subtype"], (255, 255, 255)), np.uint8)
        img[obj["mask"]] = img[obj["mask"]] * 0.5 + c * 0.5

        if obj["type"] == "box":
            pts = np.array(obj["box_points"], np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 255), 2)
        else:
            cv2.circle(img, obj["head"], 6, (255, 255, 255), -1)
            cv2.circle(img, obj["tail"], 6, (0, 0, 0), -1)
            cv2.line(img, obj["head"], obj["tail"], (255, 255, 255), 2)


# =========================
# 示例运行
# =========================

if __name__ == "__main__":

    # img = cv2.imread("./img/2025-12-15-192031.jpg")
    img = cv2.imread("./img/2025-12-25-105142.jpg")

    system = SamProduceSystem(
        sam_model_path="sam2.1_b.pt",
        edge_margin=10
    )

    result = system.run(img, visualize=True)

    print(f"SAM inference time: {result['sam_time']:.3f}s")

    for i, o in enumerate(result["objects"]):
        print(
            f"[{i}] type={o['type']} subtype={o['subtype']} center={o['center']}"
        )

    if "vis_image" in result:
        cv2.imshow("Result", result["vis_image"])
        cv2.imwrite("./result/sam_pipeline_result.jpg", result["vis_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

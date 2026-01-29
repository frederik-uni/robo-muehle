= Computer Vision
== Model Selection
For the computer vision perception stack in the robotics project, YOLOv8 was selected as the primary object detection approach because it is designed for real-time inference and is supported by a mature, integrated toolchain. In addition to strong empirical performance, YOLOv8 offers comparatively consistent training and inference. It is a lightweight variant and efficient enough for development on resource-constrained hardware. Owing to its accessible workflows and straightforward usage, this YOLO variant is also a practical choice for teams with limited prior experience in deep learning, while still providing some advanced features @ultralytics_yolov8_intro_2023. In this application context, detecting a game board and pieces, newer YOLO variants may introduce additional complexity and higher computational requirements that do not yield a proportional benefit for this required detection task. At the same time, YOLOv8 offers competitive inference speed, which is essential for responsive robotic systems that must react reliably under time constraints @ultralytics_yolov9_vs_yolov8_compare.

We implemented object detection via the Ultralytics Python library. Ultralytics provides an end-to-end pipeline that covers the full inference chain from image input to final detection, including preprocessing and postprocessing. Conceptually, YOLO (You Only Look Once) is a one-stage detector. Given an input image, it produces bounding boxes, class labels, and confidence scores within a single forward pass.

In practice, the inference workflow begins with preprocessing. The given input image is resized to a fixed resolution, and the pixel intensities are normalized. The preprocessing is followed by the forward pass through the network’s backbone, neck, and head, which yields raw detection candidates.

The backbone serves as the feature extractor that transforms raw pixels into increasingly abstract feature maps. Early backbone stages primarily encode low-level structures such as edges and textures, while deeper stages represent more complex object components. This transformation is realized via efficient convolutional blocks and progressive downsampling, which produces multiple resolution levels. The resulting multi-scale representation is critical for robust detection across object sizes, enabling the model to localize both small pieces and larger structures within the same scene. These feature maps are then propagated downstream to the remaining detection components @ultralytics_glossary_backbone.

The neck aggregates and fuses information across scales by combining high-resolution, detail-rich features with lower-resolution, semantically stronger features. As a result, the head receives enriched feature maps at multiple scales, which improves localization and classification in scenes with varying object sizes and local clutter @ultralytics_glossary_backbone.

Finally, the head generates the final predictions. YOLOv8 employs an anchor-free, decoupled head. In classical anchor-based detectors, anchors are predefined bounding-box templates of fixed sizes and aspect ratios that are tiled across the image. The model then learns offsets to adapt these templates to objects. In contrast, anchor-free detection predicts bounding boxes directly from feature-map locations without relying on predefined templates. Moreover, in YOLOv8, the head is decoupled, meaning that box regression and classification are in separate branches, which often leads to more stable optimization behavior in practice. Nevertheless, the raw output typically contains a large number of partially redundant and strongly overlapping bounding boxes @ultralytics_glossary_detection_head.

Therefore, the postprocessing stage applies confidence filtering and non-maximum suppression (short: NMS). NMS refers to a class of algorithms that selects a single representative prediction from a set of overlapping candidates. In this context, it retains the bounding box with the highest confidence score. This yields consistent final detections per object and reduces ambiguity in downstream processing @prakash_nms_learnopencv_2021.

The image detection part of this project is divided into three separate YOLOv8 models. We use separate models for stones, the board, and stacks because the target objects differ substantially in their visual characteristics. Training a single joint model would introduce unnecessary competition between these heterogeneous classes and could therefore reduce class separability, particularly under challenging conditions such as varying illumination or dense piece configurations. The board-detection module first reliably isolates the relevant region of interest and subsequently enables perspective normalization. As a result, downstream object detection becomes substantially less sensitive to camera pose and image distortion. A dedicated stack detector is justified because stacks differ in appearance from individual stones and would otherwise be more prone to class confusion and training instabilities caused by class imbalance. This modular separation also improves maintainability and iteration speed, as each model can be retrained and optimized independently.

== Dataset and Labeling Protocol
For data acquisition, we captured custom images of a Nine Men’s Morris board. To promote generalization to realistic operating conditions, we varied camera positions, viewing distance, and lighting setups, thereby introducing variance in perspective and brightness. The dataset comprises 561 images in total: 150 images for board detection, 211 images for white and black stones, and 200 images for white and black stacks. For stones and stacks, each image contains between 1 and 18 pieces.

Annotation was performed manually using bounding boxes. For board detection, the class board was labeled. For piece detection on the perspective-normalized board view, the classes are stone_white, stone_black, stack_white, and stack_black. Labeling was conducted using the browser-based tool Make Sense, which supports exporting annotations in YOLO format. For each image, a corresponding label file stores the normalized bounding-box parameters (x_center, y_center, width, height) as well as the associated class.

== Board Detection
Board detection constitutes the entry point to the perception pipeline and is executed as YOLOv8-based object detection (e.g., using models/board_best.pt or models/board_last.pt). For each frame, the detector outputs a set of candidate boxes to keep postprocessing simple and to minimize heuristic decision logic. The pipeline selects the prediction with the highest confidence (best_board_box) for subsequent processing. The result is an axis-aligned board bounding box, from which four corner points are derived as the interface to the subsequent perspective normalization stage. This normalization step reduces sensitivity to camera pose and projective distortions and thereby stabilizes downstream detection of stones and stacks by operating on a canonical, rectified board representation.

== Perspective Normalization via Homography
Reliable board-state estimation requires a consistent coordinate frame under changes in camera pose (tilt, in-plane rotation, distance). This is addressed via planar perspective normalization: once the board region is detected, it is rectified into a fixed, top-down view using a homography (projective transformation). Homographies are applicable when the observed structure is well-approximated as planar under a projective camera model @Hartley2004 @Szeliski2022. A homography maps points between two planes according to

$ p' ~ H p, quad H in RR^(3 times 3), $

where image points are written in homogeneous form  $p = (x, y, 1)^T$. The symbol $~$ denotes equality up to a non-zero scale factor. After transformation, the inhomogeneous coordinates are recovered by normalizing with the third component $w$, i.e., $x' = tilde(x) / tilde(w)$ and $y' = tilde(y) / tilde(w)$ @Hartley2004 @cmu_16385_image_homographies_slides_2024. This additional third coordinate enables translation and projective effects (e.g., foreshortening) to be expressed as a single matrix operation. This 3×3 setup and its degrees of freedom are standard in multi-view geometry @Hartley2004 @Szeliski2022.

In the implemented pipeline, board localization is performed by a learned detector that returns a bounding box. From that box, the four corners are extracted in a fixed order (top-left, top-right, bottom-right, bottom-left) and mapped to a predefined square output geometry. The homography $H$ is computed and applied with OpenCV’s perspective transform utilities (getPerspectiveTransform and warpPerspective) @opencv_geometric_image_transformations @opencv_py_geometric_transformations_tutorial. warpPerspective performs inverse-mapped, per-pixel resampling under the estimated homography, producing a rectified board image in a fixed target grid @opencv_geometric_image_transformations.

The canonical board view is rendered at a steady resolution (e.g., 1000×1000 px) so that downstream steps - index mapping, thresholding, and plausibility checks – operate in a stable pixel metric.

A design trade-off comes with estimating corners from a bounding box: it is simple and fast, but not as robust to extreme viewpoints as methods that explicitly regress corners or trace the board edge. In practice, that means the camera pose needs to stay within a limited range for reliable rectification.

== Stone/stack Detection and Candidate Extraction
Object detection is split into three parts: (1) board detection, (2) stone detection on the board, and (3) stack detection off the board. The stone and stack detectors use two classes (black and white) to directly infer color during detection. Inference is executed via the Ultralytics YOLO runtime interface, which exposes per-detection bounding boxes, class IDs, and confidence scores @ultralytics_python_usage.

A critical geometric constraint applies to stone detection: stones are detected only on the rectified board image. To prevent a mismatch between training and runtime use induced by perspective distortion, stones-model training images are warped into the same canonical geometry prior to labeling and training. This ensures that the learned appearance distribution matches the runtime input distribution (top-down board appearance).

Stack detection is performed in the original camera frame (off-board region) and therefore operates in unrectified image coordinates; no pre-warping is applied to stack training data. Stack detections are used to estimate off-board piece availability

#figure(
  image("detect.jpeg", width: 50%),
  caption: [Qualitative detection example Left: raw camera frame with board bounding box and off-board stack detections. Right: rectified top-down board view used for on-board stone detection and subsequent index mapping.],
) <fig>

Candidate extraction converts each predicted bounding box into a point hypothesis using its center $(c_x, c_y)$. Ultralytics provides box conversions including $[x_"center", y_"center", w, h]$, which enables direct retrieval of centers for subsequent mapping @ultralytics_results_reference.

Detections are filtered in two stages:

1. Minimum confidence filter: anything below a fixed confidence threshold $c_(min)$ is discarded.
2. Spatial plausibility filter (after mapping): once a detection is assigned to the nearest board index (see Section C), any detection whose center lies farther than a maximum distance $d_(max)$ from that index is rejected. This distance threshold $d_(max)$ is scaled by the estimated board grid spacing in pixels, so it stays consistent across different canonical resolutions.
If several detections land on the same board position, the best candidate wins: one hypothesis per index, usually the one with the highest confidence. If there is a tie, the closer center-to-index distance breaks it. This produces a sparse and stable set of per-position candidates suitable for discrete state reconstruction.

== Index Mapping and State Building
The normalized board view provides a deterministic mapping from image-space hypotheses to the discrete Nine Men’s Morris state. Playable positions are represented as a fixed set of indices with normalized coordinates $(x_"rel", y_"rel") in [0, 1]$ and stored in a CSV file. At runtime, those coordinates are mapped to pixel coordinates in the canonical image via $(x_i, y_i) = (x_"rel" S, y_"rel" S)$, where S is the board’s canonical side length in pixels.

Each detected stone center $(c_x, c_y)$ is assigned to the nearest index by Euclidean distance:

$ i^* = arg min_(i) sqrt((c_x - x_i)^2 + (c_y - y_i)^2). $

To increase robustness against spurious detections, a scale-aware acceptance radius is applied. This radius is based on the board geometry, e.g., the typical grid spacing, computed as the median distance to the nearest neighbor among the indices in pixel space. Detections exceeding a chosen fraction of this spacing are rejected as implausible placements.

The final output is a fixed-length occupancy array over the 24 positions using a ternary encoding compatible with the game logic: 0 for empty, −1 for black, +1 for white. In addition, temporal state tracking compares the estimated occupancy array against the previously known game state and reports a delta (changed indices) only after the same changed state is observed consistently across several consecutive frames, thereby reducing flicker-induced false updates.

== Assumptions and Limitations
Performance depends on the validity of the planar-board assumption and on accurate board localization. Errors in bounding-box localization directly affect corner derivation and therefore propagate into the homography, shifting mapped stone centers and increasing index-assignment ambiguity. @Hartley2004 @Szeliski2022 Motion blur, strong specular reflections, partial board edge occlusion, and low contrast backgrounds can cause additional degradation. These limitations motivate either tighter constraints on camera placement and lighting or the replacement of box-derived corners with corner regression/segmentation-based localization under wider viewpoint variations.

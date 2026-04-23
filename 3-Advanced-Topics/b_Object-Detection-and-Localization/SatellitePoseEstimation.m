%[text] # Satellite Pose Estimation with Deep Learning
%[text] This script demonstrates an end-to-end pipeline for estimating the pose (orientation and position) of a satellite from 2D images using deep learning. The approach uses the SPEED-UE-Cube dataset of synthetically rendered CubeSat images and combines two neural networks with classical computer vision:
%[text] 1. An *Object Detection Network (ODN)* based on YOLOv4 detects the satellite and produces a bounding box.
%[text] 2. A *Keypoint Regression Network (KRN)* predicts 11 keypoints on the cropped satellite image.
%[text] 3. A *Perspective-n-Point (PnP) solver* estimates the 6-DoF pose by comparing the predicted 2D keypoints with the known 3D wireframe model of the spacecraft. \
%[text] This script uses pre-trained models and does not require training. Participants run through the pipeline section by section to understand each stage.
%[text] Based on the [AI CubeSat Pose Estimation Workshop](https://github.com/mathworks/AI-CubeSat-Pose-Estimation-Workshop) developed in collaboration with Stanford University's Space Rendezvous Laboratory.
%%
%[text] ## References
%[text] *SPEED-UE-Cube Dataset*
%[text] \[1\] Park, T.H., Ahmed, Z., Bhattacharjee, A., Razel-Rezai, R., Graves, R., Saarela, O., Teramoto, R., Vemulapalli, K., and D'Amico, S. (2024). Spacecraft Pose Estimation Dataset of a 3U CubeSat using Unreal Engine (SPEED-UE-Cube). Available at [https://purl.stanford.edu/hw812wb1641](https://purl.stanford.edu/hw812wb1641).
%[text] \[2\] Ahmed, Z., Park, T.H., et al. SPEED-UE-Cube: A Machine Learning Dataset for Autonomous, Vision-Based Spacecraft Navigation. 46th Rocky Mountain AAS Guidance, Navigation and Control Conference, Breckenridge, Colorado, 2024.
%[text] \[3\] Stanford GitHub repository for MATLAB deep learning training example: [https://github.com/tpark94/speed-ue-cube-baseline](https://github.com/tpark94/speed-ue-cube-baseline).
%%
%[text] ## Load Data
%[text] Load the image dataset into an `imageDatastore` and the ground truth labels from a JSON file. This demo uses a subset of 15 training images from the SPEED-UE-Cube dataset.
%[text] The images are synthetically rendered views of a 3U CubeSat as seen from the camera of an approaching spacecraft. The ground truth for each image contains:
%[text] - *Position* (range): a three-element vector `r_Vo2To_vbs_true` giving the translation of the target satellite in the camera frame.
%[text] - *Orientation*: a four-element quaternion `q_vbs2tango_true` representing the rotation from the camera frame to the satellite body frame. \
%[text] The camera intrinsic parameters (focal length, principal point, distortion coefficients) are loaded from `camera.json` and converted to MATLAB format using the helper function `helper.cameraParameters_Cubesat`.
dataroot = "data";
camera = jsondecode(fileread(fullfile(dataroot, "camera.json")));

filePath = fullfile(dataroot, "train");

imds = imageDatastore(fullfile(filePath, "images"));
dataGroundTruth = jsondecode(fileread(fullfile(filePath, "train.json")));

[cameraParamMatlab, intrinsicsOpenCV] = helper.cameraParameters_Cubesat(camera);

%%
%[text] ## Load Spacecraft Wireframe Keypoints
%[text] The wireframe model defines 11 keypoints on the 3D surface of the CubeSat. These are the vertices and midpoints of the satellite body, obtained from a CAD model. The keypoints serve as the bridge between 2D image observations and 3D geometry -- they are what the keypoint regression network learns to predict, and what the PnP solver uses to recover the pose.
load(fullfile(dataroot, "cubesatPoints.mat"), "sat3dPoints");
pointsTangoBodyRef = sat3dPoints';
disp("Wireframe model loaded: " + size(pointsTangoBodyRef, 1) + " keypoints")
%%
%[text] ## Select a Sample Image
%[text] Pick one of the 15 sample images to use throughout the demo. The green crosses show the *ground truth* keypoint locations, computed by projecting the 3D wireframe vertices onto the image plane using the known pose and camera parameters.
sampleSelector = 13;

kpts_gt = helper.computeKeypointCoordinates(camera, dataGroundTruth(sampleSelector), sat3dPoints);
img = imread(fullfile(filePath, "images", dataGroundTruth(sampleSelector).filename + ".png"));

imshow(img); hold on;
scatter(kpts_gt(1,:), kpts_gt(2,:), 16, "gx", LineWidth=10);
title(["Spacecraft Image with 11 Ground Truth Keypoints"; ...
       "Projected from World to Image Frame"])
hold off
%%
%[text] ## Compute Axes Coordinates
%[text] For visualization, we project the spacecraft's body-frame coordinate axes (x, y, z) onto each image. This is done by converting the ground truth quaternion to a direction cosine matrix (DCM) and using the camera model to project the axis endpoints.
%[text] The resulting montage shows four images with overlaid axes: red = x-axis, green = y-axis, blue = z-axis.
newCoordinateAxes = zeros(size(imds.Files,1), 8);

for i = 1:length(dataGroundTruth)
    newCoordinateAxes(i,:) = helper.computeAxisCoordinates( ...
        cameraParamMatlab, ...
        dataGroundTruth(i).q_vbs2tango_true', ...
        dataGroundTruth(i).r_Vo2To_vbs_true);
end

montage({ ...
    helper.plotSpacecraftAxes(readimage(imds,1), newCoordinateAxes(1,:)), ...
    helper.plotSpacecraftAxes(readimage(imds,2), newCoordinateAxes(2,:)), ...
    helper.plotSpacecraftAxes(readimage(imds,3), newCoordinateAxes(3,:)), ...
    helper.plotSpacecraftAxes(readimage(imds,4), newCoordinateAxes(4,:))});
title("Spacecraft Images with x, y and z Axis Projection")
%%
%[text] ## Object Detection Network (ODN): Detect Spacecraft
%[text] The first stage of the pipeline locates the satellite in the image. Here we use a pre-trained YOLOv4-based object detection network. The detector outputs a bounding box around the satellite with a confidence score.
%[text] For simplicity, this network was trained offline (see [trainODN.m](https://github.com/tpark94/speed-ue-cube-baseline/blob/main/trainODN.m)). In scenarios with a uniform space background, simpler edge-detection approaches also work, but a deep learning detector generalizes to cases where the Earth appears in the background.
odn = load(fullfile("models", "odn_trained.mat"), "net");

[bbox_pr, scores, labels] = detect(odn.net, img, Threshold=0.5);
bbox_pr = double(bbox_pr);

detectedImg = insertObjectAnnotation(img, "Rectangle", bbox_pr, labels, ...
    FontSize=60, LineWidth=8);
imshow(detectedImg)
title("Spacecraft Image with Estimated Bounding Box")
%%
%[text] ## Keypoint Regression Network (KRN): Load Model
%[text] The second stage takes the cropped satellite image and predicts the 2D locations of 11 keypoints. The *Keypoint Regression Network (KRN)* is a convolutional neural network that outputs 22 values (x and y for each of the 11 keypoints), normalized to \[0, 1\] relative to the cropped image.
%[text] Like the ODN, the KRN was trained offline on the full SPEED-UE-Cube dataset (see [trainKRN.m](https://github.com/tpark94/speed-ue-cube-baseline/blob/main/trainKRN.m)). We load the pre-trained model here.
krn = load(fullfile("models", "krn_trained.mat"), "net");
krnInputSize = [224, 224];
%%
%[text] ## Crop Image to Detected Satellite
%[text] Using the bounding box from the ODN, we extract a square region of interest around the satellite. The `helper.getSquareRoI` function pads the bounding box by 20% to ensure the entire satellite is included, then the cropped region is resized to the KRN's expected 224x224 input size.
x = bbox_pr(1) + bbox_pr(3) / 2;
y = bbox_pr(2) + bbox_pr(4) / 2;
w = bbox_pr(3);
h = bbox_pr(4);
imgSize = size(img);

[xmin, ymin, xmax, ymax] = helper.getSquareRoI(x, y, w, h, imgSize([2, 1]), false);
roi = [xmin, ymin, xmax - xmin, ymax - ymin];
imgCropped = imcrop(img, roi);
imgResized = imresize(imgCropped, krnInputSize);
imgResized = im2single(imgResized);

imshow(imgResized)
title("Cropped CubeSat Image (224x224)")
%%
%[text] ## Detect Keypoints
%[text] Pass the cropped and resized image through the KRN. The network outputs normalized (x, y) coordinates for each of the 11 keypoints. We then map these back to pixel coordinates in the original full-resolution image.
%[text] The plot compares *predicted* keypoints (red) against *ground truth* (green), zoomed into the satellite region. The RMSE gives a quantitative measure of keypoint accuracy in pixels.
keypoints_pr = predict(krn.net, imgResized);

keypoints_pr = reshape(double(keypoints_pr), [2, 11]);

kpts_pr(1,:) = keypoints_pr(1,:) * (xmax - xmin) + xmin;
kpts_pr(2,:) = keypoints_pr(2,:) * (ymax - ymin) + ymin;

imshow(img); hold on
scatter(kpts_pr(1,:), kpts_pr(2,:), 16, "rx", LineWidth=10);
scatter(kpts_gt(1,:), kpts_gt(2,:), 16, "gx", LineWidth=10);
axis([xmin xmax ymin ymax])
title("Predicted vs Ground Truth Keypoints")
legend("Predicted", "Ground Truth")
hold off
%%
%[text] Compute the Root Mean Square Error (RMSE) between predicted and ground truth keypoints in pixel coordinates.
pr = reshape(kpts_pr, [], 1);
gt = reshape(kpts_gt, [], 1);
rmse = sqrt(mean((gt - pr).^2))
%%
%[text] ## Perspective-n-Point (PnP) Solver
%[text] The final stage converts the 2D keypoint predictions into a full 6-DoF pose estimate. The *PnP solver* (`estworldpose`) takes:
%[text] - The 11 predicted 2D keypoints in the image
%[text] - The corresponding 11 known 3D keypoints from the wireframe model
%[text] - The camera intrinsic parameters \
%[text] and estimates the camera's position and orientation relative to the spacecraft. This is the same algorithm used in augmented reality, robotics, and autonomous navigation to recover 3D pose from 2D-3D correspondences.
imshow(img); hold on;
scatter(kpts_pr(1,:), kpts_pr(2,:), 16, "rx", LineWidth=10);
title(["Spacecraft Image with 11 Predicted Keypoints"; ...
       "Used as Input to PnP Solver"])
hold off
%%
%[text] ## Estimate Pose and Visualize in 3D
%[text] Use the PnP solver to estimate the relative attitude and position of the camera with respect to the satellite. The 3D visualization shows the CubeSat wireframe model with the estimated camera position and viewing direction.
[PredictedSpacecraft] = estworldpose(kpts_pr', pointsTangoBodyRef, cameraParamMatlab)
dcmPredictedSpacecraft = PredictedSpacecraft.R';
rPredictedSpacecraft = PredictedSpacecraft.Translation;

pointsColorMap = [1 1 1; 1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 1 0; 1 1 0; 1 1 0; 1 1 0];
h = pcshow(pointsTangoBodyRef, pointsColorMap, ...
    VerticalAxis="Y", VerticalAxisDir="up", MarkerSize=100);
hold on
helper.plotSpacecraftWireframe(pointsTangoBodyRef);
plotCamera(Size=0.15, Orientation=dcmPredictedSpacecraft, Location=rPredictedSpacecraft);
axis equal;
hold off
set(gca, View=[-129.1886, 15.8252])
title(["Estimated Camera Position in 3D Space"; "(distance in meters)"])
%%
%[text] ## Compute Pose Error
%[text] Convert the estimated pose from the spacecraft frame to the camera frame and compare against the ground truth. The *angle error* measures the rotational difference (in degrees) between the predicted and true quaternions. The *range error* measures the absolute difference in distance (in meters) between the predicted and true camera positions.
qPredictedCamera = dcm2quat(dcmPredictedSpacecraft')';
rPredictedCamera = dcmPredictedSpacecraft * (-rPredictedSpacecraft');
qGroundtruth = dataGroundTruth(sampleSelector).q_vbs2tango_true;
rGroundtruth = dataGroundTruth(sampleSelector).r_Vo2To_vbs_true;

qError = quatmultiply(quatconj(qPredictedCamera'), qGroundtruth');
degError = 2 * atan2d(norm(qError(2:4)), qError(1));
if degError > 180
    degError = degError - 360;
end
rError = abs(norm(rPredictedCamera) - norm(rGroundtruth));

fprintf("Angle error: %.2f deg\n", degError)
fprintf("Range error: %.4f m\n", rError)
%%
%[text] ## Compare Predicted vs Ground Truth Orientation
%[text] Side-by-side comparison of the predicted and ground truth spacecraft axes overlaid on the original image. The error values are displayed on the figure.
coordPredicted = helper.computeAxisCoordinates(cameraParamMatlab, qPredictedCamera', rPredictedCamera);
coordGroundtruth = helper.computeAxisCoordinates(cameraParamMatlab, qGroundtruth', rGroundtruth);

montage({helper.plotSpacecraftAxes(img, coordPredicted), helper.plotSpacecraftAxes(img, coordGroundtruth)})
title("Predicted (left) vs Ground Truth (right) Spacecraft Axes")
text(10, 120, ["Angle Error: " + num2str(degError) + " deg"; "Range Error: " + num2str(rError) + " m"], Color=[0 1 0], BackgroundColor="k")
%%
%[text] ## Apply Pipeline to Fly-By Video
%[text] Apply the full pose estimation pipeline to a fly-by video of the CubeSat. For each sampled frame: the ODN detects the satellite, the KRN predicts 11 keypoints, and the PnP solver estimates the 6-DoF pose. The predicted keypoints and bounding box are overlaid on each frame.
%[text] The video resolution (960x600) is half the training resolution (1920x1200), so each frame is upscaled to match the camera intrinsics before processing.
v = VideoReader(fullfile(dataroot, "FlyByVideo.mp4"));
nFrames = v.NumFrames;
frameStep = 5;
frameIdx = 1:frameStep:nFrames;
nSampled = numel(frameIdx);

estimatedRange = nan(nSampled, 1);
annotatedFrames = zeros(1200, 1920, 3, nSampled, "uint8");

for k = 1:nSampled
    frame = imresize(read(v, frameIdx(k)), [1200, 1920]);

    % ODN: detect satellite bounding box
    [bbox, ~, ~] = detect(odn.net, frame, Threshold=0.5);
    if isempty(bbox)
        annotatedFrames(:,:,:,k) = frame;
        continue
    end
    bbox = double(bbox);

    % Crop to detected region
    cx = bbox(1) + bbox(3)/2;
    cy = bbox(2) + bbox(4)/2;
    [x1, y1, x2, y2] = helper.getSquareRoI(cx, cy, bbox(3), bbox(4), ...
        [size(frame,2), size(frame,1)], false);
    cropped = imcrop(frame, [x1, y1, x2-x1, y2-y1]);
    resized = im2single(imresize(cropped, krnInputSize));

    % KRN: predict keypoints
    kp = reshape(double(predict(krn.net, resized)), [2, 11]);
    kp(1,:) = kp(1,:) * (x2 - x1) + x1;
    kp(2,:) = kp(2,:) * (y2 - y1) + y1;

    % PnP: estimate pose (for range computation)
    try
        pose = estworldpose(kp', pointsTangoBodyRef, cameraParamMatlab);
        dcm = pose.R';
        r = dcm * (-pose.Translation');
        estimatedRange(k) = norm(r);
    catch
    end

    % Burn bounding box and keypoints into the image
    annotated = insertShape(frame, "filled-rectangle", bbox, ...
        Color="white", Opacity=0.15, LineWidth=3);
    annotated = insertMarker(annotated, kp', "x", Color="green", Size=15);
    annotatedFrames(:,:,:,k) = annotated;
end
%%
%[text] Play back the annotated fly-by video.
figure
h = imshow(annotatedFrames(:,:,:,1));
for k = 2:nSampled
    h.CData = annotatedFrames(:,:,:,k);
    drawnow
end
%%
%[text] ## Export Results
%[text] Uncomment the lines below to save the fly-by video as an animated GIF and export this script as an HTML report. The pre-generated report is available in the `results` folder.
if ~isfolder("results"), mkdir("results"); end
gifFile = fullfile("results", "FlyByAnnotated.gif");
for k = 1:nSampled
    [ind, cmap] = rgb2ind(annotatedFrames(:,:,:,k), 256);
    if k == 1
        imwrite(ind, cmap, gifFile, "gif", LoopCount=Inf, DelayTime=frameStep/v.FrameRate);
    else
        imwrite(ind, cmap, gifFile, "gif", WriteMode="append", DelayTime=frameStep/v.FrameRate);
    end
end
export("SatellitePoseEstimation.m", fullfile("results", "SatellitePoseEstimation.html"));
%%
%[text] *Copyright 2024-2026 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---

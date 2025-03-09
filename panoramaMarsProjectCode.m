fixedImg = imread("sol_03333_opgs_edr_ncam_NLB_693387385EDR_F0921230NCAM00259M_.JPG");
movingImg = imread("sol_03333_opgs_edr_ncam_NLB_693387301EDR_F0921230NCAM00259M_.JPG");

regOutput = registerMarsImages(movingImg,fixedImg);

showMatchedFeatures(movingImg,fixedImg,regOutput.MovingMatchedFeatures,...
        regOutput.FixedMatchedFeatures,"montage");
movingT = regOutput.Transformation;
fixedT = rigidtform2d();

% Get the image size. Both images are the same size
[nrows, ncols, ~] = size(movingImg);
[xlimMoving, ylimMoving] = outputLimits(movingT,[1 ncols],[1 nrows]);
[xlimFixed, ylimFixed] = outputLimits(fixedT,[1 ncols],[1 nrows]);

% Find the smallest and largest extents in the world coordinate system
xMin = min([xlimMoving, xlimFixed]);
xMax = max([xlimMoving, xlimFixed]);
yMin = min([ylimMoving, ylimFixed]);
yMax = max([ylimMoving, ylimFixed]);

% Width and height of panorama.
w  = round(xMax - xMin);
h = round(yMax - yMin);

% Initialize the empty panorama. This is a color image.
panorama = zeros([h, w, 3],"uint8");

blender = vision.AlphaBlender("Operation","Binary Mask","MaskSource","Input port");

panoramaView = imref2d([h w],[xMin xMax],[yMin yMax]);

movingWarped = imwarp(movingImg,movingT,"OutputView",panoramaView);
fixedWarped = imwarp(fixedImg,fixedT,"OutputView",panoramaView);
imshow(movingWarped);
imshow(fixedWarped);

mask = ones(nrows, ncols, "logical");
fixedMask = imwarp(mask, fixedT, "OutputView", panoramaView);
movingMask = imwarp(mask, movingT, "OutputView", panoramaView);

% Ensure the dimensions of the images and masks match before blending
if size(movingWarped, 3) ~= size(fixedWarped, 3)
    error("The images must have the same number of channels.");
end

if size(movingWarped, 3) == 1
    movingWarped = repmat(movingWarped, [1, 1, 3]);
    fixedWarped = repmat(fixedWarped, [1, 1, 3]);
end

if size(panorama, 3) ~= size(movingWarped, 3)
    error("The panorama and the images must have the same number of channels.");
end

% Add the first image to the panorama
panorama = blender(panorama, movingWarped, movingMask);
% Add the second image
panorama = blender(panorama, fixedWarped, fixedMask);
imshow(panorama);
panoramaGray=im2gray(panorama);
testMarsImage(panoramaGray)
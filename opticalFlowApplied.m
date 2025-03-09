% Paste your solution code from "Project: Applying Optical Flow to Detect Moving Objects" here:
%% Project: Applying Optical Flow to Detect Moving Objects
% In this problem, you will:
%% 
% * Use the Farneback method to calculate the optical flow vectors between the 
% frames and save them as a variable named |flow|.
% * Create a mask including only pixels with an optical flow vector magnitude 
% above 1. Save the result as a variable named |mask|.
% * Use image processing to update |mask| to remove regions with an area below 
% 500 pixels. 
% * Use image processing to morphologically close |mask| with a structuring 
% element of type "disk" and size 20.

frame1 = imread("Rt9Frame1.png");
frame2 = imread("Rt9Frame2.png");
montage({frame2,frame1})
%%
%Calculating the optical flow vector between frame1 & frame2

myOpticalFlow = opticalFlowFarneback;
estimateFlow(myOpticalFlow,im2gray(frame1));
flow = estimateFlow(myOpticalFlow,im2gray(frame2)); 
% myOpticalFlow stores the previous frame
%%
%Just to see

imshow(frame2)
hold on
plot(flow,"DecimationFactor",[15 15],"ScaleFactor",7)
hold off
%%
%Creating Mask with optical flow magnitude 1+

vm = flow.Magnitude;
maskThreshold = 1.0;
mask = (vm(:,:)>maskThreshold);
%image processing 

se = strel("disk",20,0);
mask = bwareafilt(mask, [500, inf]);
mask = imclose(mask, se);


% Uncomment below to view your optical flow vectors on your masked image
maskedFrame = frame2; maskedFrame(repmat(~mask, [1 1 3])) = 0;
imshow(frame2)
hold on
plot(flow,"DecimationFactor",[15 15],"ScaleFactor",7)
hold off
figure
imshow(maskedFrame)

% Write new code to count the number of cars moving in each direction and their x-direction velocities
%Here, you will use optical flow and this mask to determine how many cars are traveling in each direction and their velocity.
%In this problem, you will:
%Copy your code from the previous problem to calculate the optical flow and create a flow magnitude mask.
%Calculate the average  for each object in the mask.
%Store the average  for every object moving to the left and has an average  above 3 in a variable named leftVx. These are your cars driving left.
%Store the average  for every object moving to the right and has an average  above 3 in a variable named rightVx. These are your cars driving right.
%Store the total number of cars driving to the left as numCarsLeft.
%Store the total number of cars driving to the right as numCarsRight.
%You will be graded based on the following:
%Correctly detecting the number of left-moving vehicles, numCarsLeft
%Correctly detecting the number of right-moving vehicles, numCarsRight
%Correct values for the array of left-moving velocities, leftVx
%Correct values for the array of right-moving velocities, rightVx

%% New Code: Counting Cars Moving in Each Direction
% We now use the optical flow (particularly the x-component, flow.Vx) along with the mask
% to determine:
%  - The average x-direction velocity for each connected region in the mask.
%  - Which objects (cars) are moving left (negative Vx) or right (positive Vx),
%    considering only those objects whose average |Vx| > 3.

% Compute region properties from the mask using flow.Vx as the intensity image.
stats = regionprops(mask, flow.Vx, 'MeanIntensity');

% Initialize arrays to store the average x-velocity for left- and right-moving vehicles.
leftVx = [];    % Will store average Vx for vehicles moving left (negative values)
rightVx = [];   % Will store average Vx for vehicles moving right (positive values)

% Process each detected object in the mask.
for i = 1:length(stats)
    avgVx = stats(i).MeanIntensity;
    % Only consider objects with sufficient average horizontal speed (absolute value > 3)
    if abs(avgVx) > 3  
         if avgVx < 0
            leftVx = [leftVx, avgVx];
        elseif avgVx > 0
            rightVx = [rightVx, avgVx];
        end
    end
end

% Count the number of vehicles moving in each direction.
numCarsLeft = length(leftVx);
numCarsRight = length(rightVx);

% Display results.
fprintf('Number of cars moving left: %d\n', numCarsLeft);
fprintf('Number of cars moving right: %d\n', numCarsRight);
fprintf('Left-moving vehicles average Vx:\n');
disp(leftVx);
fprintf('Right-moving vehicles average Vx:\n');
disp(rightVx);

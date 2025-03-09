% Task 1: Prepare images for classification and split into training and testing sets
% Create an image datastore from the Roadside Ground Cover images.
% Make you you select your select path to labelled folder accurately
% For this example two labelled folders are used Snow & No Snow
imds = imageDatastore(fullfile('D:\Faraz\cv_matlab_cert\Computer_Vision\Computer Vision for Engineering and Science\Data\Data\MathWorks Images\Roadside Ground Cover'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the datastore into training and testing subsets.
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.85, 'randomized');

% Count the number of images labeled as "Snow" in the training datastore.
snowTrainCount = sum(imdsTrain.Labels == 'Snow');

% Display the count.
fprintf('Number of images labeled as "Snow" in the training datastore: %d\n', snowTrainCount);

% Optional: display the number of snow images in the test set.
snowTestCount = sum(imdsTest.Labels == 'Snow');
fprintf('Number of images labeled as "Snow" in the testing datastore: %d\n', snowTestCount);

% Optional: display the total number of images in each set.
fprintf('Number of images in the training datastore: %d\n', numel(imdsTrain.Labels));
fprintf('Number of images in the testing datastore: %d\n', numel(imdsTest.Labels));

% Task 2: Extract saturation-based features from the images

% Loop through each image in the training set
numTrainImages = numel(imdsTrain.Files);
trainFeatures = zeros(numTrainImages, 1); % Initialize an array to hold the features

for i = 1:numTrainImages
    % Read the image
    img = readimage(imdsTrain, i);
    
    % Convert the image to HSV color space
    imgHSV = rgb2hsv(img);
    
    % Extract the saturation plane
    imgSaturation = imgHSV(:,:,2);
    
    % Calculate the mean saturation value and store it as a feature
    trainFeatures(i) = mean(imgSaturation(:));
end

% Loop through each image in the test set
numTestImages = numel(imdsTest.Files);
testFeatures = zeros(numTestImages, 1); % Initialize an array to hold the features

for i = 1:numTestImages
    % Read the image
    img = readimage(imdsTest, i);
    
    % Convert the image to HSV color space
    imgHSV = rgb2hsv(img);
    
    % Extract the saturation plane
    imgSaturation = imgHSV(:,:,2);
    
    % Calculate the mean saturation value and store it as a feature
    testFeatures(i) = mean(imgSaturation(:));
end

% Save the features and labels to a table for use with the Classification Learner app
trainData = table(trainFeatures, imdsTrain.Labels, 'VariableNames', {'Saturation', 'Label'});
testData = table(testFeatures, imdsTest.Labels, 'VariableNames', {'Saturation', 'Label'});

% Save the tables to files for later use
save('trainData.mat', 'trainData');
save('testData.mat', 'testData');

fprintf('Feature extraction and data preparation complete. Train and test data saved to trainData.mat and testData.mat.\n');

% Now you are ready to use Classification learner App in matlab
% Run the scipt after making relevant changes and open in app
% Use the default 5-fold cross-validation.
% train & test the model 
 
% Step 1: Prepare a traffic sign data set for classification and extracting predictor features.
dataFolder = fullfile('D:\Faraz\cv_matlab_cert\Computer_Vision\Computer Vision for Engineering and Science\Data\Data\Traffic Signs');
imds = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Step 2: Preparing Your Data by creating an image datastore
% Labels are assigned based on the subfolder names, which correspond to the traffic sign type.
labels = countEachLabel(imds);
disp(labels);

% Step 3: Split into Training and Test Sets
% Create randomized training and test sets. 80% of the images to be assigned to the training set.
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.80, 'randomized');

% Step 4: Use bagOfFeatures to create a set of predictor features
bag = bagOfFeatures(imdsTrain);

% Step 5: Train classification models using the MATLAB Classification Learner App
% and the predictor features you have just created. Your goal is to train a model with a test accuracy of at least 90%
% Encode the training images.
trainFeatures = encode(bag, imdsTrain);
trainLabels = imdsTrain.Labels;

% Save the encoded features and labels to a table for use in the Classification Learner App.
trainData = array2table(trainFeatures);
trainData.Labels = trainLabels;

% Step 5(a): Launch Classification Learner App
% To launch the app, uncomment the following line:
% classificationLearner

% Step 6: Extract features from test sets and test data
testFeatures = encode(bag, imdsTest);
testLabels = imdsTest.Labels;

% Save the encoded test features and labels to a table for evaluation.
testData = array2table(testFeatures);
testData.Labels = testLabels;

% Save the training and test data tables to files for later use.
save('trainData.mat', 'trainData');
save('testData.mat', 'testData');

fprintf('Data preparation complete. Training and test data saved to trainData.mat and testData.mat.\n');
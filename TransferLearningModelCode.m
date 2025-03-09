%% Data Preparation
% Define the folder paths for Train and Test data
trainFolder = 'D:\Faraz\cv_matlab_cert\Deep Learning for Computer Vision\Data\MathWorks Created\ASL Alphabet\Classification\Train';
testFolder  = 'D:\Faraz\cv_matlab_cert\Deep Learning for Computer Vision\Data\MathWorks Created\ASL Alphabet\Classification\Test';

% Create an imageDatastore for the training folder (with labels from folder names)
trainDatastoreFull = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the full training datastore into 80% training and 20% validation
[imdsTrain, imdsVal] = splitEachLabel(trainDatastoreFull, 0.8, 'randomized');

% Create an imageDatastore for the test folder
imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Define the input image size required by ResNet‑18 ([height width channels])
inputSize = [224 224 3];

% Create augmented image datastores that automatically resize images,
% while preserving labels for training and validation.
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsVal   = augmentedImageDatastore(inputSize, imdsVal);
augimdsTest  = augmentedImageDatastore(inputSize, imdsTest);

% Display dataset statistics using the underlying imageDatastores
disp('Data preparation complete:');
disp(['Number of training images: ', num2str(numel(imdsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imdsVal.Files))]);
disp(['Number of test images: ', num2str(numel(imdsTest.Files))]);

%% Network Modification: Transfer Learning with ResNet‑18
% Load the pre‐trained ResNet‐18 network
net = resnet18;

% Convert to a layer graph to allow modifications
lgraph = layerGraph(net);

% Remove the original final layers which are specific to ImageNet (1000 classes)
layersToRemove = {'fc1000','prob','ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

% Determine the number of classes in your ASL dataset
numClasses = numel(categories(trainDatastoreFull.Labels));

% Create new layers for our ASL classification task:
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
       'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')
    ];

% Add the new layers to the layer graph
lgraph = addLayers(lgraph, newLayers);

% Connect the new fully connected layer to the network.
% For ResNet-18, the output of the last pooling layer is named 'pool5'.
lgraph = connectLayers(lgraph, 'pool5', 'new_fc');

% (Optional) Visualize the final network architecture:
% analyzeNetwork(lgraph);

%% Training Options and Model Training
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network using the augmented training datastore and modified network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

disp('Training complete.');

%% Model Evaluation
% Classify the test images using the trained network
predictedLabels = classify(trainedNet, augimdsTest);

% Retrieve the true labels from the test datastore
trueLabels = imdsTest.Labels;

% Calculate test accuracy
testAccuracy = sum(predictedLabels == trueLabels) / numel(trueLabels) * 100;
disp(['Test Accuracy: ', num2str(testAccuracy), '%']);

% Plot a confusion matrix for a detailed view of classification performance
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix');

% Visualize some test predictions: Display 9 test images with their predicted labels
figure;
for i = 1:9
    subplot(3,3,i);
    % Use the original imdsTest to read the image (full resolution)
    img = readimage(imdsTest, i);
    label = predictedLabels(i);
    imshow(img);
    title(char(label));
end

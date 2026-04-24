%[text] # Deep Learning In 6 Lines of Code
%[text] ## Load Pre-trained CNN
%[text] Pre-trained CNNs are deep networks that are already trained on thousands of different images. They can work for classification "right out of the box." 
%[text] There are several networks available from our [Add-Ons](https://www.mathworks.com/add-ons/ALEXNET/) section under the home tab. We will use a pretrained model called [AlexNet](https://en.wikipedia.org/wiki/AlexNet), which is trained on a subsection of the [ImageNet](http://www.image-net.org/) database. It classifies images of 1000 different categories. 
% Load the AlexNet neural network
[net, classNames] = imagePretrainedNetwork("alexnet");

% View the network layers and connections
analyzeNetwork(net)
%%
%[text] ## Classify Image of "Peppers"
% Load in an image: 'peppers.png'. Hint: help imread
%[ADD CODE HERE]

% Show the figure on the screen. Hint: help imshow
%[ADD CODE HERE]

% Resize the image for AlexNet requirements [227 227]. Hint: help imresize
%[ADD CODE HERE]

% Classify what the image is based on the AlexNet pretrained network
% Can you guess the function you need to classify your image?
X = single(imResized); % Convert to single datatype for prediction
%[ADD CODE HERE]
%%
%[text] ## Export Results
%[text] Uncomment the lines below to export this script as an HTML report.
%if ~isfolder("results"), mkdir("results"); end
%export("GettingStarted.m", fullfile("results", "GettingStarted.html"));
%%
%[text] *Copyright 2024-2026 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---

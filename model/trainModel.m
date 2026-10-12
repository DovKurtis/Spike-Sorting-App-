function [model, modelInfo] = trainModel(Xtrain, Ytrain, modelFile, modelType)

modelType = lower(string(modelType));

switch modelType

    case "linear"

        X = double(Xtrain);
        Y = categorical(Ytrain);

        if exist(modelFile, 'file')

            S = load(modelFile, 'model', 'modelInfo');

            if ~strcmpi(S.modelInfo.ModelType, 'linear')
                error('Existing model is not a linear model.');
            end

            model = fit(S.model, X, Y);
            modelInfo = S.modelInfo;

        else

            model = fitclinear(X, Y, ...
                'Learner', 'logistic', ...
                'Regularization', 'ridge', ...
                'ClassNames', categorical([0 1]));

            model = incrementalLearner(model);

            modelInfo = struct();
            modelInfo.CreatedAt = datestr(now);

        end


    case "cnn"

        X = single(Xtrain);

        % CNN labels
        Y = categorical(Ytrain, [0 1], {'noise','spike'});

        % dlnetwork sequence input:
        % each cell = [timeSamples x 1 channel]
        Xseq = cell(size(X,1),1);

        for i = 1:size(X,1)
            Xseq{i} = X(i,:).';
        end

        if exist(modelFile, 'file')

            S = load(modelFile, 'model', 'modelInfo');

            if ~strcmpi(S.modelInfo.ModelType, 'cnn')
                error('Existing model is not a CNN model.');
            end

            model = S.model;
            modelInfo = S.modelInfo;

        else

            layers = [

                sequenceInputLayer(1, ...
                Normalization="none", ...
                MinLength=size(Xtrain,2))

                convolution1dLayer(15, 32, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(2, Stride=2)

                convolution1dLayer(11, 64, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(2, Stride=2)

                convolution1dLayer(7, 128, Padding="same")
                batchNormalizationLayer
                reluLayer

                globalAveragePooling1dLayer

                fullyConnectedLayer(2)
                softmaxLayer
            ];

            model = dlnetwork(layers);

            modelInfo = struct();
            modelInfo.CreatedAt = datestr(now);

        end

        options = trainingOptions("adam", ...
            InitialLearnRate=1e-3, ...
            MaxEpochs=8, ...
            MiniBatchSize=256, ...
            Shuffle="every-epoch", ...
            Verbose=false, ...
            Plots="none");

        model = trainnet( ...
            Xseq, ...
            Y, ...
            model, ...
            "crossentropy", ...
            options);


    otherwise

        error('modelType must be ''linear'' or ''cnn''.');

end


%% COMMON MODEL INFO

modelInfo.ModelType = char(modelType);

if modelType == "linear"
    modelInfo.Type = 'linear logistic waveform-only';
else
    modelInfo.Type = '1D CNN waveform-only';
end

modelInfo.LastUpdatedAt = datestr(now);
modelInfo.LastBatchSize = size(Xtrain,1);

modelInfo.LabelMeaning = '0=noise, 1=spike';
modelInfo.Normalization = 'none';

save(modelFile, 'model', 'modelInfo');

end
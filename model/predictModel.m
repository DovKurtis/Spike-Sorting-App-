function [pred, score] = predictModel(W, modelFile)

persistent model modelInfo loadedModelFile

modelFile = char(modelFile);

if isempty(model) || isempty(loadedModelFile) || ...
        ~strcmp(modelFile, loadedModelFile)

    if ~isfile(modelFile)
        error('Selected model does not exist:\n%s', modelFile);
    end

    S = load(modelFile, 'model', 'modelInfo');

    if ~isfield(S, 'model')
        error('Selected file does not contain a variable named "model".');
    end

    if ~isfield(S, 'modelInfo') || ~isfield(S.modelInfo, 'ModelType')
        error('Selected model does not contain modelInfo.ModelType.');
    end

    model = S.model;
    modelInfo = S.modelInfo;
    loadedModelFile = modelFile;

    fprintf('Loaded %s model: %s\n', modelInfo.ModelType, modelFile);
end


switch lower(string(modelInfo.ModelType))

    case "linear"

        [pred, score] = predict(model, double(W));


    case "cnn"

        X = single(W);

        % Each row of W is one waveform.
        % CNN expects one sequence per cell: [time x channels].
        Xseq = cell(size(X,1),1);

        for i = 1:size(X,1)
            Xseq{i} = X(i,:).';
        end

        % Returns N x 2 scores:
        % column 1 = noise
        % column 2 = spike
        score = minibatchpredict( ...
            model, ...
            Xseq, ...
            MiniBatchSize=256);

        classes = categorical({'noise','spike'});

        pred = scores2label(score, classes);


    otherwise

        error('Unknown model type: %s', modelInfo.ModelType);

end

end
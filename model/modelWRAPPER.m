% modelWRAPPER.m
% THE ONLY PLACE YOU HAVE TO EDIT OR INTERACT WITH ANYTHING.2

clear; clc;
NelsonLabDatPath = 'path to/ data';

modelPath = 'path to /model';

subjNum = 17;
appendStr = 'kilosort_NS17_cnn_Aligned'; 
constrain = '_Thresh0.mat'; %only train on AVs with this in the name — use Thresh0 for kilosort-only, set = '' for all  
modelType = 'cnn';   % 'linear' or 'cnn'
cnnSamplesPerClassPerFile = 3000;
rng(172);
%%
% for each loaded rawsortdat (which depends on your above options) it
% finds, 1 of 4 thigns happen: 

% Exact path + exact creation time exists in manifest
% → unchanged
% → do not load it
% 
% Session/channel does not exist anywhere in manifest
% → brand-new session/channel
% → load only that file
% → incrementally train existing model on only that new data
% 
% Session/channel exists in manifest, BUT current newest file has a different
% path and/or creation timestamp
% → replacement
% → old model contains obsolete data
% → load the newest file for EVERY session/channel
% → retrain the entire model from scratch
% → replace the manifest completely with the current newest-file manifest

% 4th thing: one of the manifests (ids of the rawsortdats used to make the training set) was not found
% again in your files, ie you deleted it —> it will retrain again from
% scracth — the behavior is the same as if you ahd changed one file. it
% retrains completely in order to not have the old (non existent rather than
% 'different') data 

%% dont change 
appendStr = regexprep(appendStr, '[^A-Za-z0-9_-]', '_');

if ~exist(modelPath, 'dir')
    mkdir(modelPath);
end

if ~isempty(appendStr) && appendStr(1) ~= '_'
    appendStr = ['_' appendStr];
end
%% get subject's data to append to XYtrainAll  

D = dir(fullfile(NelsonLabDatPath,'Intracranial data','Raw Neural Data per Subj',sprintf('NS%d Raw Neural Data',subjNum),sprintf('NS%d Neuralynx Sess*',subjNum)));
nlynxFoldSessPaths = fullfile(string({D.folder}), string({D.name}))';


XYtrainAll = fullfile(modelPath, ['XYtrainAll' appendStr '.mat']);

% Read ONLY the old manifest. Does not load Xtrain/Ytrain.
if exist(XYtrainAll, 'file')
    M = matfile(XYtrainAll);

    if any(strcmp(who(M), 'sourceManifest'))
        oldManifest = M.sourceManifest;
    else
        oldManifest = strings(0,1);
    end
else
    oldManifest = strings(0,1);
end

%% SCAN FILE METADATA FIRST — DO NOT LOAD WAVEFORMS YET

allPaths = strings(0,1);
allIDs = strings(0,1);
allCreated = NaT(0,1);

for iSess = 1:numel(nlynxFoldSessPaths)

    iSessRawSortDatDir = dir(fullfile( ...
        nlynxFoldSessPaths(iSess), ...
        sprintf('RawSortDat_NS*%s', constrain)));

    iSessPaths = fullfile( ...
        string({iSessRawSortDatDir.folder}), ...
        string({iSessRawSortDatDir.name}))';

    for k = 1:numel(iSessPaths)

        currentPath = iSessPaths(k);

        tok = regexp(char(currentPath), ...
            'NS(\d+)_sess(\d+)_ch(\d+)', ...
            'tokens', 'once');

        if isempty(tok)
            error('Could not identify subject/session/channel from: %s', currentPath);
        end

        currentID = "NS" + tok{1} + "_sess" + tok{2} + "_ch" + tok{3};

        p = java.nio.file.Paths.get( ...
            char(currentPath), ...
            javaArray('java.lang.String',0));

        attrs = java.nio.file.Files.readAttributes( ...
            p, ...
            java.lang.Class.forName( ...
                'java.nio.file.attribute.BasicFileAttributes'), ...
            javaArray('java.nio.file.LinkOption',0));

        createdMillis = attrs.creationTime().toMillis();

        currentCreated = datetime( ...
            double(createdMillis)/1000, ...
            'ConvertFrom','posixtime');

        allPaths(end+1,1) = currentPath;
        allIDs(end+1,1) = currentID;
        allCreated(end+1,1) = currentCreated;
    end
end


%% KEEP ONLY NEWEST FILE FOR EACH SUBJECT/SESSION/CHANNEL

currentPaths = strings(0,1);
currentIDs = strings(0,1);
currentCreated = NaT(0,1);

uniqueIDs = unique(allIDs);

for k = 1:numel(uniqueIDs)

    candidates = find(allIDs == uniqueIDs(k));

    [~, newestLocal] = max(allCreated(candidates));
    winner = candidates(newestLocal);

    currentPaths(end+1,1) = allPaths(winner);
    currentIDs(end+1,1) = allIDs(winner);
    currentCreated(end+1,1) = allCreated(winner);
end


%% BUILD CURRENT MANIFEST

currentManifest = strings(numel(currentPaths),1);

for k = 1:numel(currentPaths)

    dt = currentCreated(k);
    dt.Format = 'yyyy-MM-dd HH:mm:ss';

    currentManifest(k) = ...
        currentPaths(k) + " | " + string(dt);
end


%% EXTRACT SUBJECT/SESSION/CHANNEL IDs FROM OLD MANIFEST

oldIDs = strings(numel(oldManifest),1);

for k = 1:numel(oldManifest)

    tok = regexp(char(oldManifest(k)), ...
        'NS(\d+)_sess(\d+)_ch(\d+)', ...
        'tokens', 'once');

    if isempty(tok)
        error('Could not identify subject/session/channel from old manifest: %s', ...
            oldManifest(k));
    end

    oldIDs(k) = "NS" + tok{1} + "_sess" + tok{2} + "_ch" + tok{3};
end


%% DECIDE WHAT KIND OF TRAINING IS REQUIRED

isNew = false(numel(currentManifest),1);
isReplacement = false(numel(currentManifest),1);
isUnchanged = false(numel(currentManifest),1);

for k = 1:numel(currentManifest)

    oldIdx = find(oldIDs == currentIDs(k), 1);

    if isempty(oldIdx)

        % Never trained this subject/session/channel
        isNew(k) = true;

    elseif oldManifest(oldIdx) == currentManifest(k)

        % Exact same file + exact same creation timestamp
        isUnchanged(k) = true;

    else

        % Same subject/session/channel, but source file changed
        isReplacement(k) = true;
    end
end

% Detect previously-trained session/channels that no longer exist at all
isRemoved = ~ismember(oldIDs, currentIDs);


fprintf('\nCURRENT TRAINING STATUS\n');
fprintf('New:         %d\n', sum(isNew));
fprintf('Replacement: %d\n', sum(isReplacement));
fprintf('Unchanged:   %d\n', sum(isUnchanged));
fprintf('Removed:     %d\n', sum(isRemoved));

%% MODEL FILE

modelFile = fullfile(modelPath, ['model' appendStr '.mat']);


%% CASE 1: NOTHING CHANGED

firstRun = ~exist(XYtrainAll, 'file') || ~exist(modelFile, 'file');

if ~firstRun && ~any(isNew) && ~any(isReplacement) && ~any(isRemoved)
    fprintf('\nNothing changed. Model unchanged.\n');
    return

end


%% CASE 2: REPLACEMENT EXISTS
% Old model contains obsolete training data.
% Rebuild BOTH training archive and model from current newest files.
% IMPORTANT: files are processed ONE AT A TIME — no giant Xtrain in RAM.


if firstRun || any(isReplacement) || any(isRemoved) || ...
        (strcmpi(modelType,'cnn') && any(isNew))

    if firstRun
        fprintf('\nFirst run. Building model from current newest files.\n');
    else
        fprintf('\nTraining sources changed. Rebuilding model from current newest files.\n');
    end

    if exist(modelFile, 'file')
        delete(modelFile);
    end

    if exist(XYtrainAll, 'file')
        delete(XYtrainAll);
    end

    nArchiveRows = 0;

    initX = [];
    initY = [];
    modelInitialized = false;

    cnnX = single([]);
    cnnY = zeros(0,1);

    cnnSampleFile = strings(0,1);
    cnnSampleRow = zeros(0,1);

    for k = 1:numel(currentPaths)

        currentFile = currentPaths(k);

        fprintf('\n[FULL RETRAIN %d/%d]\n%s\n', ...
            k, numel(currentPaths), currentFile);

        rsd = load(currentFile);

        if ~isfield(rsd, 'FrameData')
            error('No FrameData in selected training file: %s', currentFile);
        end

        Xbatch = rsd.FrameData.Wfs;
        Ybatch = rsd.FrameData.Status;
        Ybatch(Ybatch ~= 0,:) = 1;

        fprintf('Waveforms: %d | Noise: %d | Non-noise: %d\n', ...
            size(Xbatch,1), ...
            sum(Ybatch == 0), ...
            sum(Ybatch ~= 0));


        %% CNN: randomly sample this file for the final mixed training set
        if strcmpi(modelType, 'cnn')

            noiseIdx = find(Ybatch == 0);
            spikeIdx = find(Ybatch == 1);

            nNoiseTake = min(cnnSamplesPerClassPerFile, numel(noiseIdx));
            nSpikeTake = min(cnnSamplesPerClassPerFile, numel(spikeIdx));

            if nNoiseTake > 0
                noiseTake = noiseIdx(randperm(numel(noiseIdx), nNoiseTake));
            else
                noiseTake = zeros(0,1);
            end

            if nSpikeTake > 0
                spikeTake = spikeIdx(randperm(numel(spikeIdx), nSpikeTake));
            else
                spikeTake = zeros(0,1);
            end

            takeRows = [noiseTake(:); spikeTake(:)];

            cnnX = [cnnX; single(Xbatch(takeRows,:))];
            cnnY = [cnnY; Ybatch(takeRows)];

            cnnSampleFile = [cnnSampleFile; ...
                repmat(string(currentFile), numel(takeRows), 1)];

            cnnSampleRow = [cnnSampleRow; takeRows];

            fprintf('CNN sampled: %d noise | %d spike\n', ...
                nNoiseTake, nSpikeTake);

        end


        % Initialize only after both classes 0 and 1 have been seen.
        if strcmpi(modelType, 'linear')

            % Linear model can still train file-by-file incrementally.
            if ~modelInitialized

                initX = [initX; Xbatch];
                initY = [initY; Ybatch];

                if numel(unique(initY)) == 2
                    [model, modelInfo] = trainModel( ...
                        initX, initY, modelFile, modelType);

                    modelInitialized = true;

                    clear initX initY
                end

            else

                [model, modelInfo] = trainModel( ...
                    Xbatch, Ybatch, modelFile, modelType);

            end

        end

        % Rebuild XYtrainAll without holding historical data in RAM.
        if k == 1

            Xtrain = Xbatch;
            Ytrain = Ybatch;

            save(XYtrainAll, ...
                'Xtrain', 'Ytrain', '-v7.3');

            nArchiveRows = size(Xbatch,1);

            clear Xtrain Ytrain

        else

            M = matfile(XYtrainAll, 'Writable', true);

            nNew = size(Xbatch,1);

            M.Xtrain( ...
                nArchiveRows+1:nArchiveRows+nNew, ...
                1:size(Xbatch,2)) = Xbatch;

            M.Ytrain( ...
                nArchiveRows+1:nArchiveRows+nNew, ...
                1) = Ybatch;

            nArchiveRows = nArchiveRows + nNew;
        end

        clear rsd Xbatch Ybatch
    end

    if strcmpi(modelType, 'linear') && ~modelInitialized
        error('Cannot train model: the complete training set does not contain both noise (0) and spike (1) examples.');
    end

    if strcmpi(modelType, 'cnn')

        if isempty(cnnY) || ~all(ismember([0 1], unique(cnnY)))
            error('CNN training sample does not contain both noise and spike examples.');
        end

        fprintf('\nCNN MIXED SAMPLE BEFORE FINAL BALANCE\n');
        fprintf('Noise: %d | Spike: %d\n', ...
            sum(cnnY == 0), sum(cnnY == 1));

        % Globally balance the two classes.
        noiseRows = find(cnnY == 0);
        spikeRows = find(cnnY == 1);

        nPerClass = min(numel(noiseRows), numel(spikeRows));

        noiseRows = noiseRows(randperm(numel(noiseRows), nPerClass));
        spikeRows = spikeRows(randperm(numel(spikeRows), nPerClass));

        keepRows = [noiseRows; spikeRows];

        % Shuffle noise and spike examples together.
        keepRows = keepRows(randperm(numel(keepRows)));

        cnnX = cnnX(keepRows,:);
        cnnY = cnnY(keepRows);

        cnnSampleFile = cnnSampleFile(keepRows);
        cnnSampleRow = cnnSampleRow(keepRows);

        fprintf('CNN FINAL TRAINING SET: %d waveforms | %d noise | %d spike\n', ...
            numel(cnnY), sum(cnnY == 0), sum(cnnY == 1));

        [model, modelInfo] = trainModel( ...
            cnnX, cnnY, modelFile, modelType);

        % Save exact provenance of every waveform actually used by CNN.
        modelInfo.CNNSampleFile = cnnSampleFile;
        modelInfo.CNNSampleRow = cnnSampleRow;
        modelInfo.CNNSamplesPerClassPerFile = cnnSamplesPerClassPerFile;

        save(modelFile, 'model', 'modelInfo');

    end

    % Manifest now represents EXACTLY what rebuilt model/archive used.
    M = matfile(XYtrainAll, 'Writable', true);
    M.sourceManifest = currentManifest;

    fprintf('\nFull retrain complete.\n');
    return
end


%% CASE 3: ONLY BRAND-NEW SUBJECT/SESSION/CHANNEL FILES
% Existing model remains valid.
% Train ONLY on genuinely new files.

newIdx = find(isNew);

fprintf('\nIncremental update: %d new files.\n', numel(newIdx));

M = matfile(XYtrainAll, 'Writable', true);
sz = size(M, 'Xtrain');
nArchiveRows = sz(1);

for j = 1:numel(newIdx)

    k = newIdx(j);
    currentFile = currentPaths(k);

    fprintf('\n[NEW TRAINING FILE %d/%d]\n%s\n', ...
        j, numel(newIdx), currentFile);

    rsd = load(currentFile);

    if ~isfield(rsd, 'FrameData')
        error('No FrameData in selected training file: %s', currentFile);
    end

    Xbatch = rsd.FrameData.Wfs;
    Ybatch = rsd.FrameData.Status;
    Ybatch(Ybatch ~= 0,:) = 1;

    fprintf('Waveforms: %d | Noise: %d | Non-noise: %d\n', ...
        size(Xbatch,1), ...
        sum(Ybatch == 0), ...
        sum(Ybatch ~= 0));


    % CASE 3 is linear-only. CNN source changes trigger a full mixed retrain above.
    [model, modelInfo] = trainModel(Xbatch, Ybatch, modelFile, 'linear');

    % Append only this file's waveforms to archive.
    nNew = size(Xbatch,1);

    M.Xtrain( ...
        nArchiveRows+1:nArchiveRows+nNew, ...
        1:size(Xbatch,2)) = Xbatch;

    M.Ytrain( ...
        nArchiveRows+1:nArchiveRows+nNew, ...
        1) = Ybatch;

    nArchiveRows = nArchiveRows + nNew;

    clear rsd Xbatch Ybatch
end

% Archive/model now include old sources + new sources.
M.sourceManifest = currentManifest;

fprintf('\nIncremental update complete.\n');
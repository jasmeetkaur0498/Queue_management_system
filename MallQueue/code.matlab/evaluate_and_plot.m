function evaluate_and_plot(resultFiles)
if nargin < 1
    files = dir('results/*.mat');
    resultFiles = arrayfun(@(f) fullfile('results',f.name), files, 'UniformOutput', false);
end

summary = [];
names = {};
for i=1:length(resultFiles)
    s = load(resultFiles{i});
    if isfield(s,'episodeAvgWait')
        avgWait = mean(s.episodeAvgWait(max(1,end-9):end));
        avgStepR = mean(s.episodeAvgStepReward(max(1,end-9):end));
        name = resultFiles{i};
    elseif isfield(s,'avgWait')
        avgWait = mean(s.avgWait(max(1,end-9):end));
        avgStepR = mean(s.episodeReward(max(1,end-9):end)) / 200;
        name = resultFiles{i};
    else
        continue;
    end
    summary(end+1,:) = [avgWait, avgStepR]; %#ok<AGROW>
    names{end+1} = name; %#ok<AGROW>
    fprintf('File: %s | avgWait(last10)=%.3f | avgStepReward(last10)=%.3f\n', name, avgWait, avgStepR);
end

T = array2table(summary,'VariableNames',{'avgWait','avgStepReward'});
if ~exist('results','dir'), mkdir('results'); end
outname = fullfile('results',['summary_' datestr(now,'yyyymmdd_HHMMSS') '.csv']);
writetable(T,outname);
fprintf('Summary saved to %s\n', outname);

% If at least two results present, print comparison baseline vs first RL
if size(summary,1) >= 2
    baselineWait = summary(1,1);
    rlWait = summary(2,1);
    improv = (baselineWait - rlWait)/baselineWait * 100;
    fprintf('Baseline avgWait=%.3f | RL avgWait=%.3f | Improvement=%.2f%%\n', baselineWait, rlWait, improv);
end
end


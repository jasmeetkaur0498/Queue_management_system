function outFile = qlearning_train(env, params)
% QLEARNING_TRAIN - robust Q-learning trainer for MallEnv (restricted actions)
% Action space: 1..C (prioritize counter i), C+1 = do nothing.
% This avoids destructive 'close counter' actions that permanently reduce capacity.

if ~isfield(params,'alpha'), params.alpha = 0.1; end
if ~isfield(params,'gamma'), params.gamma = 0.95; end
if ~isfield(params,'epsilon'), params.epsilon = 1.0; end
if ~isfield(params,'epsilon_min'), params.epsilon_min = 0.05; end
if ~isfield(params,'epsilon_decay'), params.epsilon_decay = 0.995; end
if ~isfield(params,'maxEpisodes'), params.maxEpisodes = 300; end
if ~isfield(params,'maxSteps'), params.maxSteps = 200; end
if ~isfield(params,'numBuckets'), params.numBuckets = 20; end
if ~isfield(params,'seed'), params.seed = 1; end

rng(params.seed);
isObj = isa(env,'MallEnv');

% get initial observation
if isObj
    try r = env.reset(); if isa(r,'MallEnv'), env = r; obs = env.observe(); else obs = r; end
    catch, obs = env.observe(); end
else
    if isfield(env,'reset') && isa(env.reset,'function_handle'), env = env.reset(); end
    obs = [env.state, sum(env.active), mean(env.state)];
end

obsLen = numel(obs);
C = obsLen - 2;

% RESTRICTED ACTION SPACE: 1..C = prioritize counter i; C+1 = do nothing
numActions = C + 1;

numBuckets = params.numBuckets;
Q = zeros(numBuckets, numActions);

episodeReward = zeros(1, params.maxEpisodes);
episodeAvgStepReward = zeros(1, params.maxEpisodes);
episodeAvgWait = zeros(1, params.maxEpisodes);

for ep = 1:params.maxEpisodes
    % reset environment
    if isObj
        try r = env.reset(); if isa(r,'MallEnv'), env = r; obs = env.observe(); else obs = r; end
        catch, obs = env.observe(); end
    else
        if isfield(env,'reset') && isa(env.reset,'function_handle'), env = env.reset(); else env.state=zeros(1,C); env.active=ones(1,C); end
        obs = [env.state, sum(env.active), mean(env.state)];
    end

    totalR = 0; stepCount = 0; lastObs = obs;
    for t = 1:params.maxSteps
        stepCount = stepCount + 1;
        avgQ = lastObs(end);
        bucket = min(numBuckets, max(1, 1 + floor(avgQ / (env.maxQueueLength / numBuckets))));

        % epsilon-greedy
        if rand < params.epsilon
            action = randi(numActions);
        else
            [~, action] = max(Q(bucket,:));
        end

        % Map action to env: 1..C => prioritize; C+1 => do nothing
        mappedAction = action;
        if mappedAction > C
            mappedAction = 2*C + 1; % env's "do nothing" index remains the same internally
        end

        % Step environment robustly
        if isObj
            try [newObs, rwd, done, env] = env.step(mappedAction); catch [newObs, rwd, done] = env.step(mappedAction); end
        else
            if exist('env_step','file') == 2, [newObs, rwd, done, env] = env_step(env, mappedAction); else error('qlearning_train:NoEnvStep','No env.step available'); end
        end

        totalR = totalR + rwd;
        lastObs = newObs;

        % next bucket
        nextBucket = min(numBuckets, max(1, 1 + floor(lastObs(end) / (env.maxQueueLength / numBuckets))));

        % Q update
        Q(bucket, action) = Q(bucket, action) + params.alpha * ( rwd + params.gamma * max(Q(nextBucket,:)) - Q(bucket, action) );

        if exist('done','var') && done, break; end
    end

    episodeReward(ep) = totalR;
    episodeAvgStepReward(ep) = totalR / max(1, stepCount);
    episodeAvgWait(ep) = lastObs(end);

    % decay epsilon (consider slower decay if learning is unstable)
    params.epsilon = max(params.epsilon_min, params.epsilon * params.epsilon_decay);

    if mod(ep,20)==0 || ep==1
        fprintf('Episode %d/%d - avgStepR=%.3f avgWait=%.3f eps=%.3f\n', ep, params.maxEpisodes, episodeAvgStepReward(ep), episodeAvgWait(ep), params.epsilon);
    end
end

% Save
if ~exist('results','dir'), mkdir('results'); end
tstamp = datestr(now,'yyyymmdd_HHMMSS');
outFile = fullfile('results', ['qlearn_' tstamp '.mat']);
save(outFile, 'episodeReward', 'episodeAvgStepReward', 'episodeAvgWait', 'Q', 'params');
fprintf('Q-learning results saved to %s\n', outFile);
end




function outFile = baseline_eval(env, params)
if nargin < 2, params = struct(); end
if ~isfield(params,'maxEpisodes'), params.maxEpisodes = 200; end
if ~isfield(params,'maxSteps'), params.maxSteps = 200; end
if ~isfield(params,'seed'), params.seed = 1; end

isObj = isa(env, 'MallEnv');
rng(params.seed);
episodeReward = zeros(1, params.maxEpisodes);
episodeAvgWait = zeros(1, params.maxEpisodes);

for ep = 1:params.maxEpisodes
    if isObj
        try r = env.reset(); if isa(r,'MallEnv'), env = r; obs = env.observe(); else obs = r; end; catch, obs = env.observe(); end
    else
        if isfield(env,'reset') && isa(env.reset,'function_handle'), env = env.reset(); else env.state=zeros(1,env.numCounters); env.active=ones(1,env.numCounters); end
        obs = [env.state, sum(env.active), mean(env.state)];
    end

    totalR = 0; lastObs = obs;
    for t = 1:params.maxSteps
        if isObj
            try curObs = env.observe(); catch, curObs = [env.state, sum(env.active), mean(env.state)]; end
        else curObs = [env.state, sum(env.active), mean(env.state)]; end
        C = length(curObs)-2;
        queueLens = curObs(1:C);
        [~, idx] = max(queueLens);
        if queueLens(idx) == 0, action = 2*C+1; else action = idx; end

        if isObj
            try [newObs, r, done, env] = env.step(action); catch [newObs, r, done] = env.step(action); end
        else
            if exist('env_step','file') == 2, [newObs, r, done, env] = env_step(env, action); else error('No env.step for struct env'); end
        end

        totalR = totalR + r;
        lastObs = newObs;
        if exist('done','var') && done, break; end
    end

    episodeReward(ep) = totalR;
    episodeAvgWait(ep) = lastObs(end);
    if mod(ep,20)==0 || ep==1, fprintf('Baseline Episode %d/%d - totalReward=%.2f avgWait=%.2f\n', ep, params.maxEpisodes, totalR, episodeAvgWait(ep)); end
end

if ~exist('results','dir'), mkdir('results'); end
tstamp = datestr(now,'yyyymmdd_HHMMSS');
outFile = fullfile('results',['baseline_' tstamp '.mat']);
save(outFile,'episodeReward','episodeAvgWait','params');
fprintf('Baseline saved to %s\n', outFile);
end



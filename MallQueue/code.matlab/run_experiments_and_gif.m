%% run_experiments_and_gif.m
% Runs baseline + Q-learning across seeds and saves CSVs + demo GIFs for seed 1.

clear; clc;
data = preprocess_data('queue_dataoriginal.csv');

seeds = [1,11,101];
results = [];

for i=1:length(seeds)
    s = seeds(i);
    params = struct('numCounters',6,'maxQueueLength',30,'timeStepMinutes',5,'seed',s,'arrivalScale',0.25,'serviceScale',1.2,'verbose',false);
    env = MallEnv(data, params);

    bfile = baseline_eval(env, struct('maxEpisodes',200,'maxSteps',200,'seed',s));
    qfile = qlearning_train(env, struct('alpha',0.1,'gamma',0.95,'epsilon',1.0,'epsilon_decay',0.995,'epsilon_min',0.05,...
        'maxEpisodes',300,'maxSteps',200,'numBuckets',20,'seed',s));

    B = load(bfile); Q = load(qfile);
    baseline_last = mean(B.episodeAvgWait(max(1,end-9):end));
    q_last       = mean(Q.episodeAvgWait(max(1,end-9):end));
    improvement  = (baseline_last - q_last)/baseline_last * 100;
    results(end+1,:) = [s baseline_last q_last improvement]; %#ok<AGROW>

    % optionally create gifs for seed 1 only
    if i == 1
        % create env for visual demo
        env_demo = MallEnv(data, params);
        % create baseline gif
        create_policy_gif(env_demo, bfile, 'demo_baseline.gif', 120);
        % create learned-policy gif (greedy)
        create_policy_gif(env_demo, qfile, 'demo_qlearn_greedy.gif', 120);
    end
end

T = array2table(results, 'VariableNames', {'seed','baseline_avgWait','q_avgWait','improvement_pct'});
outcsv = fullfile('results', ['multi_seed_summary_' datestr(now,'yyyymmdd_HHMMSS') '.csv']);
writetable(T,outcsv);
fprintf('Saved multi-seed summary to %s\n', outcsv);

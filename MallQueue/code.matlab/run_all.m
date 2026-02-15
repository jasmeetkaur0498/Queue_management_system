clear; clc;
data = preprocess_data('queue_dataoriginal.csv');

params = struct('numCounters',6,'maxQueueLength',30,'timeStepMinutes',5,'seed',42,'arrivalScale',0.7,'serviceScale',1.2,'verbose',false);
env = MallEnv(data, params);

% baseline (short)
bfile = baseline_eval(env, struct('maxEpisodes',200,'maxSteps',200,'seed',params.seed));

% qlearning (short then full)
qfile_short = qlearning_train(env, struct('alpha',0.1,'gamma',0.95,'epsilon',1.0,'epsilon_decay',0.995,'epsilon_min',0.05,'maxEpisodes',100,'maxSteps',100,'numBuckets',20,'seed',params.seed));
qfile_full  = qlearning_train(env, struct('alpha',0.1,'gamma',0.95,'epsilon',1.0,'epsilon_decay',0.995,'epsilon_min',0.05,'maxEpisodes',300,'maxSteps',200,'numBuckets',20,'seed',params.seed));

evaluate_and_plot({bfile, qfile_short, qfile_full});

function outFile = dqn_train(env, params)
% dqn_train - MATLAB RL Toolbox DQN training (requires RL Toolbox)
% This is a template — you will need RL Toolbox.

% Prepare observation spec and action spec
obsDim = env.numCounters + 2;
obsInfo = rlNumericSpec([obsDim 1]);
actInfo = rlFiniteSetSpec(1:(2*env.numCounters+1));

% create function env wrappers: user must implement rlFunctionEnv step/reset wrappers
% For simplicity, this template assumes you will create wrappers or use a custom env.

error('DQN training requires RL Toolbox and an rlFunctionEnv wrapper. See comments in file.');
end

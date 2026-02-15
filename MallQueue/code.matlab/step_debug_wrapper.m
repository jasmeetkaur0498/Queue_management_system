function step_debug_wrapper(env, action, steps)
% simple wrapper: runs env.step(action) steps times and prints net change
if nargin < 3, steps = 10; end
C = env.numCounters;
for t = 1:steps
    prev_sum = sum(env.state);
    [obs, r, done, env] = env.step(action);
    post_sum = sum(env.state);
    net = post_sum - prev_sum;
    fprintf('t=%2d: netChange=%+d  avgWait=%.3f  reward=%.3f  done=%d\n', t, net, obs(end), r, done);
    if done
        fprintf('→ env signalled done at step %d\n', t);
        break;
    end
end
end

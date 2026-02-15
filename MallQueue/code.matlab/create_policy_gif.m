function create_policy_gif(env, resultMatFile, outGifName, maxSteps)
% Creates a bar-chart GIF for one episode using the policy from resultMatFile.
% If resultMatFile is baseline file, we run the baseline heuristic. If it's qlearn file,
% we run greedy policy using saved Q table (bucketed).

if nargin<4, maxSteps = 200; end
s = load(resultMatFile);

% prepare Q and params if available
if isfield(s,'Q')
    Q = s.Q;
else
    Q = [];
end

% reset env
obs = env.reset();
C = env.numCounters;

% prepare figure
h = figure('Visible','off','Position',[100 100 640 360]);
axis tight manual
filename = outGifName;

for t = 1:maxSteps
    % decide action
    if isempty(Q)
        % baseline heuristic: prioritize longest queue
        [~, idx] = max(obs(1:C));
        if obs(idx)==0
            mappedAction = 2*C+1; % do nothing
        else
            mappedAction = idx;
        end
    else
        % greedy from Q: compute bucket from avgWait
        numBuckets = size(Q,1);
        bucket = min(numBuckets, max(1, 1 + floor(obs(end) / (env.maxQueueLength/numBuckets))));
        [~, actIdx] = max(Q(bucket,:));
        % our training used reduced action-space mapping:
        if actIdx > C
            mappedAction = 2*C + 1;
        else
            mappedAction = actIdx;
        end
    end

    [obs, r, done, env] = env.step(mappedAction);

    % draw bar chart of queues
    bar(obs(1:C));
    ylim([0 env.maxQueueLength]);
    title(sprintf('%s | t=%d | avgWait=%.2f', outGifName, t, obs(end)));
    xlabel('Counter'); ylabel('Queue length');
    drawnow;

    % capture frame
    frame = getframe(h);
    im = frame2im(frame);
    [A,map] = rgb2ind(im,256);
    if t == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.15);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.15);
    end

    if done, break; end
end
close(h);
fprintf('Saved GIF: %s\n', filename);
end
